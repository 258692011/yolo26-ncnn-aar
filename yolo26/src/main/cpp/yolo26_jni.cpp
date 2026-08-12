// YOLO26 + NCNN Android 推理引擎（AAR）
// 模型: ultralytics yolo export model=yolo26n.pt format=ncnn 导出的 .param/.bin
// 输出约定（导出图内已含 DFL 解码 + sigmoid）: [1, 4+nc, N]，通道0-3为解码后的 xywh(640坐标系)，4..为类别分数
// 若模型输出为原始 DFL (64+nc) 通道（如 ncnn 官方示例的改造导出），自动回退到 C++ DFL 解码
#include <jni.h>
#include <android/bitmap.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "net.h"

#define TAG "yolo26"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

struct Detection {
    int label;
    float score;
    float x1, y1, x2, y2;
};

struct OutView {
    const float* data = nullptr;
    int anchors = 0;   // 锚点数（通常 8400）
    int channels = 0;  // 通道数（4+nc 或 64+nc）
    bool anchorsOnW = true;  // true: out.w=anchors, out.h=channels（内存 channel-major）
                              // false: out.w=channels, out.h=anchors（每锚点连续）

    inline float at(int anchor, int channel) const {
        return anchorsOnW ? data[channel * anchors + anchor] : data[anchor * channels + channel];
    }
};

inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

inline float intersection_area(const Detection& a, const Detection& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    if (ix2 <= ix1 || iy2 <= iy1) return 0.f;
    return (ix2 - ix1) * (iy2 - iy1);
}

void qsort_descent_inplace(std::vector<Detection>& v, int l, int r) {
    int i = l, j = r;
    float p = v[(l + r) / 2].score;
    while (i <= j) {
        while (v[i].score > p) i++;
        while (v[j].score < p) j--;
        if (i <= j) { std::swap(v[i], v[j]); i++; j--; }
    }
    if (l < j) qsort_descent_inplace(v, l, j);
    if (i < r) qsort_descent_inplace(v, i, r);
}

std::vector<int> nms(const std::vector<Detection>& dets, float thresh) {
    std::vector<int> picked;
    std::vector<float> areas(dets.size());
    for (size_t i = 0; i < dets.size(); i++)
        areas[i] = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);
    for (size_t i = 0; i < dets.size(); i++) {
        bool keep = true;
        for (int j : picked) {
            if (dets[i].label != dets[j].label) continue;
            float inter = intersection_area(dets[i], dets[j]);
            float uni = areas[i] + areas[j] - inter;
            if (inter / uni > thresh) { keep = false; break; }
        }
        if (keep) picked.push_back((int)i);
    }
    return picked;
}

bool parse_io_names(const std::string& paramText, std::string& inputName, std::vector<std::string>& outputNames) {
    inputName.clear();
    outputNames.clear();
    std::istringstream iss(paramText);
    std::string line;
    std::getline(iss, line);  // 首行: 层数 blob数
    std::set<std::string> produced, consumed;
    bool firstNonInput = true;
    while (std::getline(iss, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string type, name;
        int nb = 0, nt = 0;
        if (!(ls >> type >> name >> nb >> nt)) continue;
        std::vector<std::string> bottoms, tops;
        std::string s;
        for (int k = 0; k < nb && (ls >> s); k++) bottoms.push_back(s);
        for (int k = 0; k < nt && (ls >> s); k++) tops.push_back(s);
        if (type == "Input") {
            if (inputName.empty() && !tops.empty()) inputName = tops[0];
            for (auto& t : tops) produced.insert(t);
            continue;
        }
        if (firstNonInput && !bottoms.empty()) {
            if (inputName.empty()) inputName = bottoms[0];
            firstNonInput = false;
        }
        for (auto& b : bottoms) consumed.insert(b);
        for (auto& t : tops) produced.insert(t);
    }
    for (auto& p : produced)
        if (!consumed.count(p)) outputNames.push_back(p);
    return !outputNames.empty();
}

std::string read_asset_text(AAssetManager* mgr, const char* name) {
    AAsset* a = AAssetManager_open(mgr, name, AASSET_MODE_BUFFER);
    if (!a) return "";
    size_t len = (size_t)AAsset_getLength(a);
    std::string s(len, '\0');
    if (len > 0) AAsset_read(a, &s[0], len);
    AAsset_close(a);
    return s;
}

// 从 param 文本自动推导模型输入尺寸:
// pnnx 导出的头部分含形如 "Reshape xxx 1 1 in out 0=G 1=G 2=128" 的层,
// G = 输入尺寸/32 (stride-32 特征图网格, 640 导出为 20, 320 导出为 10)
int parse_input_size(const std::string& paramText) {
    std::istringstream iss(paramText);
    std::string line;
    std::getline(iss, line);  // 首行计数
    int best = 0;
    while (std::getline(iss, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.compare(0, 8, "Reshape ") != 0) continue;
        std::istringstream ls(line);
        std::string type, name;
        int nb = 0, nt = 0;
        if (!(ls >> type >> name >> nb >> nt)) continue;
        std::string s;
        for (int i = 0; i < nb + nt && (ls >> s); i++) {}
        int w = 0, h = 0;
        std::string tok;
        while (ls >> tok) {
            if (tok.size() >= 2 && tok[1] == '=') {
                int v = atoi(tok.c_str() + 2);
                if (tok[0] == '0') w = v;
                else if (tok[0] == '1') h = v;
            }
        }
        if (w > 0 && w == h) {
            int sz = w * 32;
            if (sz >= 96 && sz <= 2048 && sz > best) best = sz;
        }
    }
    return best;  // 0 = 未识别, 调用方回退默认 640
}

class Yolo26Engine {
public:
    ~Yolo26Engine() { net_.clear(); }

    bool load(AAssetManager* mgr, const char* paramAsset, const char* binAsset,
              const char* paramPath, const char* binPath, int inputSize) {
        mgr_ = mgr;
        inputSize_ = 640;  // 默认; 若显式传入或可从 param 解析则覆盖
        paramText_.clear();
        int rp = -1, rb = -1;
        if (paramPath != nullptr && binPath != nullptr) {
            rp = net_.load_param(paramPath);
            rb = net_.load_model(binPath);
            std::ifstream f(paramPath, std::ios::binary);
            if (f) paramText_ = std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        } else if (mgr_ != nullptr && paramAsset != nullptr && binAsset != nullptr) {
            rp = net_.load_param(mgr_, paramAsset);
            rb = net_.load_model(mgr_, binAsset);
            paramText_ = read_asset_text(mgr_, paramAsset);
        }
        if (rp != 0 || rb != 0) {
            LOGE("load failed: param=%d model=%d", rp, rb);
            return false;
        }
        // 输入尺寸: 显式传入优先; 否则从 param 的 Reshape 网格自动推导 (0 或非法时回退 640)
        if (inputSize > 0) {
            inputSize_ = inputSize;
        } else {
            int parsed = parse_input_size(paramText_);
            if (parsed > 0) inputSize_ = parsed;
        }
        LOGI("input size = %d (explicit=%d)", inputSize_, inputSize);
        if (!parse_io_names(paramText_, inputName_, outputNames_)) {
            LOGE("parse_io_names failed");
            return false;
        }
        // 推理性能: 保持 ncnn 默认 (全核自适应); 实验证明硬编码线程/绑核均无增益, 且跨设备最稳
        LOGI("opt: threads=%d", net_.opt.num_threads);
        LOGI("input=%s outputs=%zu first=%s", inputName_.c_str(), outputNames_.size(),
             outputNames_.empty() ? "-" : outputNames_[0].c_str());
        return true;
    }

    std::vector<Detection> detect(const unsigned char* rgba, int imgW, int imgH,
                                  float confThresh, float nmsThresh) {
        std::vector<Detection> dets;
        if (rgba == nullptr || imgW <= 0 || imgH <= 0) return dets;

        // ---- letterbox 到 inputSize_ x inputSize_ ----
        // 注意: 静态导出模型固定输入 NxN, 必须精确补到 inputSize_（不能只对齐 32 的倍数）
        const int target = inputSize_;
        float scale;
        int w, h;
        if (imgW > imgH) { scale = (float)target / imgW; w = target; h = (int)(imgH * scale); }
        else { scale = (float)target / imgH; h = target; w = (int)(imgW * scale); }
        int wpad = target - w;
        int hpad = target - h;

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgba, ncnn::Mat::PIXEL_RGBA2RGB, imgW, imgH, w, h);
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                               ncnn::BORDER_CONSTANT, 114.f);
        const float norm[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
        in_pad.substract_mean_normalize(nullptr, norm);

        // ---- 推理 ----
        ncnn::Extractor ex = net_.create_extractor();
        ex.input(inputName_.c_str(), in_pad);
        std::vector<ncnn::Mat> outs;
        for (const auto& name : outputNames_) {
            ncnn::Mat out;
            int ret = ex.extract(name.c_str(), out);
            if (ret != 0) { LOGE("extract %s failed %d", name.c_str(), ret); return dets; }
            outs.push_back(out);
        }

        // ---- 解码 ----
        std::vector<Detection> proposals;
        for (const auto& out : outs) {
            decode_output(out, scale, wpad, hpad, confThresh, proposals);
        }
        if (proposals.empty()) return dets;

        qsort_descent_inplace(proposals, 0, (int)proposals.size() - 1);
        auto picked = nms(proposals, nmsThresh);
        for (int i : picked) dets.push_back(proposals[i]);
        return dets;
    }

private:
    void decode_output(const ncnn::Mat& out, float scale, int wpad, int hpad,
                       float confThresh, std::vector<Detection>& proposals) {
        if (out.empty()) return;
        const float* data = (const float*)out.data;
        OutView view;
        view.data = data;
        // 通用布局判定: 通道数总是较小维（4+nc 或 64+nc），锚点数总是较大维
        // 640 导出: (w=8400, h=84); 320 导出: (w=2100, h=84); ncnn 示例风格: (w=144, h=8400)
        view.channels = std::min(out.w, out.h);
        view.anchors = std::max(out.w, out.h);
        view.anchorsOnW = (out.w == view.anchors);

        if (view.channels > 100) {
            // 原始 DFL 输出: 64+nc 通道，需要 C++ 侧 DFL softmax 解码（参考 ncnn examples/yolo11.cpp）
            decode_raw_dfl(view, scale, wpad, hpad, confThresh, proposals);
        } else {
            decode_decoded(view, scale, wpad, hpad, confThresh, proposals);
        }
    }

    // 图内已解码: 通道0-3 = cx,cy,w,h（640 填充坐标系），4.. = 已 sigmoid 类别分数
    void decode_decoded(const OutView& v, float scale, int wpad, int hpad,
                        float confThresh, std::vector<Detection>& proposals) {
        const int nc = v.channels - 4;
        for (int a = 0; a < v.anchors; a++) {
            float cx = v.at(a, 0), cy = v.at(a, 1), bw = v.at(a, 2), bh = v.at(a, 3);
            if (bw <= 0.f || bh <= 0.f) continue;
            float best = -FLT_MAX;
            int bestL = -1;
            for (int c = 0; c < nc; c++) {
                float s = v.at(a, 4 + c);
                if (s > best) { best = s; bestL = c; }
            }
            if (bestL < 0 || best < confThresh) continue;
            float x0 = (cx - bw * 0.5f - wpad / 2) / scale;
            float y0 = (cy - bh * 0.5f - hpad / 2) / scale;
            float x1 = (cx + bw * 0.5f - wpad / 2) / scale;
            float y1 = (cy + bh * 0.5f - hpad / 2) / scale;
            proposals.push_back({bestL, best, x0, y0, x1, y1});
        }
    }

    // 原始 DFL: 通道 0..63 = 4 边 × 16 bins（未 softmax），64.. = 未 sigmoid 类别分数
    void decode_raw_dfl(const OutView& v, float scale, int wpad, int hpad,
                        float confThresh, std::vector<Detection>& proposals) {
        constexpr int regMax = 16;
        const int nc = v.channels - regMax * 4;
        for (int a = 0; a < v.anchors; a++) {
            float best = -FLT_MAX;
            int bestL = -1;
            for (int c = 0; c < nc; c++) {
                float s = sigmoid(v.at(a, regMax * 4 + c));
                if (s > best) { best = s; bestL = c; }
            }
            if (bestL < 0 || best < confThresh) continue;
            float dist[4];
            for (int k = 0; k < 4; k++) {
                float row[regMax];
                float m = -FLT_MAX;
                for (int l = 0; l < regMax; l++) { row[l] = v.at(a, k * regMax + l); m = std::max(m, row[l]); }
                float sum = 0.f, acc = 0.f;
                for (int l = 0; l < regMax; l++) { float e = expf(row[l] - m); row[l] = e; sum += e; }
                for (int l = 0; l < regMax; l++) acc += l * row[l];
                dist[k] = acc / sum;
            }
            // 锚点所属 stride 与网格坐标（按输入尺寸推导，支持任意 imgsz 导出）
            const int grid8 = inputSize_ / 8, grid16 = inputSize_ / 16, grid32 = inputSize_ / 32;
            const int seg8 = grid8 * grid8, seg16 = grid16 * grid16;
            int stride, grid, idx = a;
            if (a < seg8) { stride = 8; grid = grid8; }
            else if (a < seg8 + seg16) { stride = 16; grid = grid16; idx = a - seg8; }
            else { stride = 32; grid = grid32; idx = a - seg8 - seg16; }
            int gx = idx % grid;
            int gy = idx / grid;
            float cx = (gx + 0.5f) * stride;
            float cy = (gy + 0.5f) * stride;
            float x0 = (cx - dist[0] * stride - wpad / 2) / scale;
            float y0 = (cy - dist[1] * stride - hpad / 2) / scale;
            float x1 = (cx + dist[2] * stride - wpad / 2) / scale;
            float y1 = (cy + dist[3] * stride - hpad / 2) / scale;
            proposals.push_back({bestL, best, x0, y0, x1, y1});
        }
    }

    ncnn::Net net_;
    AAssetManager* mgr_ = nullptr;
    std::string paramText_;
    std::string inputName_ = "in0";
    std::vector<std::string> outputNames_;
    int inputSize_ = 640;
};

// ---------- JNI ----------

Yolo26Engine* from_handle(jlong h) { return reinterpret_cast<Yolo26Engine*>(h); }

jlong nativeCreate(JNIEnv* env, jobject, jobject assetManager, jstring paramAsset, jstring binAsset,
                   jstring paramPath, jstring binPath, jint inputSize) {
    AAssetManager* mgr = assetManager ? AAssetManager_fromJava(env, assetManager) : nullptr;
    const char* pa = paramAsset ? env->GetStringUTFChars(paramAsset, nullptr) : nullptr;
    const char* ba = binAsset ? env->GetStringUTFChars(binAsset, nullptr) : nullptr;
    const char* pp = paramPath ? env->GetStringUTFChars(paramPath, nullptr) : nullptr;
    const char* bp = binPath ? env->GetStringUTFChars(binPath, nullptr) : nullptr;

    Yolo26Engine* eng = new Yolo26Engine();
    bool ok = eng->load(mgr, pa, ba, pp, bp, inputSize);

    if (pa) env->ReleaseStringUTFChars(paramAsset, pa);
    if (ba) env->ReleaseStringUTFChars(binAsset, ba);
    if (pp) env->ReleaseStringUTFChars(paramPath, pp);
    if (bp) env->ReleaseStringUTFChars(binPath, bp);

    if (!ok) { delete eng; return 0; }
    return reinterpret_cast<jlong>(eng);
}

jobjectArray nativeDetect(JNIEnv* env, jobject, jlong handle, jobject bitmap,
                          jfloat confThresh, jfloat nmsThresh) {
    Yolo26Engine* eng = from_handle(handle);
    if (!eng) return nullptr;

    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) return nullptr;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return nullptr;

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) return nullptr;
    std::vector<Detection> dets = eng->detect((const unsigned char*)pixels, info.width, info.height,
                                              confThresh, nmsThresh);
    AndroidBitmap_unlockPixels(env, bitmap);

    jclass detCls = env->FindClass("com/yolo26/ncnn/Detection");
    if (!detCls) return nullptr;
    jmethodID ctor = env->GetMethodID(detCls, "<init>", "(IFFFFF)V");
    jobjectArray arr = env->NewObjectArray((jsize)dets.size(), detCls, nullptr);
    for (size_t i = 0; i < dets.size(); i++) {
        jobject d = env->NewObject(detCls, ctor, (jint)dets[i].label, (jfloat)dets[i].score,
                                   (jfloat)dets[i].x1, (jfloat)dets[i].y1,
                                   (jfloat)dets[i].x2, (jfloat)dets[i].y2);
        env->SetObjectArrayElement(arr, (jsize)i, d);
        env->DeleteLocalRef(d);
    }
    env->DeleteLocalRef(detCls);
    return arr;
}

void nativeClose(JNIEnv*, jobject, jlong handle) {
    if (handle != 0) delete from_handle(handle);
}

jstring nativeVersion(JNIEnv* env, jclass) {
    std::string v = "ncnn " NCNN_VERSION_STRING;
    return env->NewStringUTF(v.c_str());
}

}  // namespace

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;
    jclass cls = env->FindClass("com/yolo26/ncnn/Yolo26NcnnDetector");
    if (!cls) return JNI_ERR;
    static const JNINativeMethod methods[] = {
        {"nativeCreate",  "(Landroid/content/res/AssetManager;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;I)J",
         (void*)&nativeCreate},
        {"nativeDetect",  "(JLandroid/graphics/Bitmap;FF)[Lcom/yolo26/ncnn/Detection;",
         (void*)&nativeDetect},
        {"nativeClose",   "(J)V", (void*)&nativeClose},
        {"nativeVersion", "()Ljava/lang/String;", (void*)&nativeVersion},
    };
    if (env->RegisterNatives(cls, methods, 4) != JNI_OK) return JNI_ERR;
    env->DeleteLocalRef(cls);
    return JNI_VERSION_1_6;
}
