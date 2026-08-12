package com.yolo26.ncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

/**
 * YOLO26 + NCNN 安卓检测器（实例式 API，兼容 AutoJS DexClassLoader 用法）。
 *
 * 用法（Android）:
 *   Yolo26NcnnDetector yolo = new Yolo26NcnnDetector();
 *   if (!yolo.init(paramPath, binPath, 0)) { ...失败... }   // inputSize=0 自动识别
 *   Detection[] dets = yolo.detect(bitmap, 0.25f, 0.45f);
 *   yolo.close();
 *
 * 用法（AutoJS，与萌新 ncnn 插件同构）:
 *   var yolo = clazz.newInstance();
 *   yolo.init(param, bin, 0);   // inputSize=0 自动识别
 *   var dets = yolo.detect(bitmap, 0.25, 0.45);
 *   yolo.close();
 */
public class Yolo26NcnnDetector {

    static {
        System.loadLibrary("yolo26");
    }

    private long handle = 0;

    public Yolo26NcnnDetector() {
    }

    /**
     * 从文件路径加载模型。
     *
     * @param paramPath  .param 文件绝对路径
     * @param binPath    .bin 文件绝对路径
     * @param inputSize  模型输入尺寸；0 = 从 .param 自动识别（推荐）
     * @return 是否初始化成功
     */
    public boolean init(String paramPath, String binPath, int inputSize) {
        close();
        handle = nativeCreate(null, null, null, paramPath, binPath, inputSize);
        return handle != 0;
    }

    /**
     * 从 assets 加载模型（AAR 内置模型时使用）。
     */
    public boolean init(AssetManager assetManager, String paramAsset, String binAsset, int inputSize) {
        close();
        handle = nativeCreate(assetManager, paramAsset, binAsset, null, null, inputSize);
        return handle != 0;
    }

    /**
     * 推理一帧，返回按置信度降序、已做 NMS 的检测。坐标相对原图，未裁剪。
     */
    public Detection[] detect(Bitmap bitmap, float confThreshold, float nmsThreshold) {
        if (handle == 0 || bitmap == null) {
            return new Detection[0];
        }
        Bitmap bmp = bitmap;
        if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
            bmp = bitmap.copy(Bitmap.Config.ARGB_8888, false);
        }
        return nativeDetect(handle, bmp, confThreshold, nmsThreshold);
    }

    /** 释放 native 资源。 */
    public void close() {
        if (handle != 0) {
            nativeClose(handle);
            handle = 0;
        }
    }

    /** ncnn 版本信息（调试用）。 */
    public static String version() {
        return nativeVersion();
    }

    private static native String nativeVersion();

    private native long nativeCreate(AssetManager assetManager, String paramAsset, String binAsset,
                                     String paramPath, String binPath, int inputSize);

    private native Detection[] nativeDetect(long h, Bitmap bitmap, float confThreshold, float nmsThreshold);

    private native void nativeClose(long h);
}
