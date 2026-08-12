# yolo26-ncnn-aar

YOLO26 (Ultralytics) 检测模型的 Android AAR 封装，基于 Tencent ncnn 推理引擎。

## 产物与验证状态

| 项目 | 说明 |
|---|---|
| 模型 | `yolo26n` (COCO 80 类)，`yolo export format=ncnn` 官方导出，fp16 |
| 验证 | 与 ultralytics 官方 ncnn 预测逐框对比一致（bus.jpg: bus + 4 persons，坐标/置信度吻合） |
| ncnn | 20260526 官方预编译 `libncnn.so`（CPU，arm64-v8a / armeabi-v7a / x86_64 / x86） |
| 后处理 | 图内已含 DFL 解码 + sigmoid，App 侧仅做 letterbox 逆变换 + 类别 NMS |

## 快速开始（使用方 App）

> **模型不内置在 AAR**（AAR 仅含 Java 类 + 动态库，约 17MB）。模型文件由调用方自行提供：
> 用 `yolo export model=yolo26n.pt format=ncnn quantize='fp16'` 导出得到 `yolo26n_ncnn_model.param/.bin`（fp16，约 5MB），
> 放进 App 的 assets 或私有目录。换 320/480/640 等任意 imgsz 导出的模型均可用（`inputSize=0` 自动识别）。

```kotlin
// 1. 将 yolo26.aar 放入 app/libs/，build.gradle.kts 添加:
//    implementation(files("libs/yolo26.aar"))

// 2. 使用（实例式 API，纯 Java 无 Kotlin 依赖）
val yolo = Yolo26NcnnDetector()
// 方式 A: 从 assets 加载
if (!yolo.init(applicationContext.assets, "yolo26n_ncnn_model.param", "yolo26n_ncnn_model.bin", 0)) {
    // 初始化失败
}
// 方式 B: 从文件路径加载（模型已拷贝到 App 私有目录等）
// yolo.init("/data/.../yolo26n_ncnn_model.param", "/data/.../yolo26n_ncnn_model.bin", 0)

val dets = yolo.detect(bitmap, 0.25f, 0.45f)   // 任意尺寸 Bitmap
for (d in dets) {
    Log.i("YOLO26", "class=${d.label} score=${d.score} box=(${d.x1},${d.y1},${d.x2},${d.y2})")
}
yolo.close()
```

## AutoJS 用法（桌面 yolo26ncnn/ 文件夹）

AutoJS 不能直接用 AAR，已把 Java API 转成 dex + .so 的同构插件（用法与萌新 ncnn 一致）：

```js
var yolo26 = require("yolo26ncnn.js");
var yolo = yolo26.yolo26("yolo26n_ncnn_model");           // 自动识别输入尺寸
var dets = yolo.detect(img.getBitmap(), 0.25, 0.45);      // (bitmap, conf, nms)
// d.getLabel() / d.getScore() / d.getX1()/getY1()/getX2()/getY2()
yolo.close();
```

文件夹结构：`yolo26ncnn/`（js + dex + lib/<abi>/*.so + model/*.param/.bin）整体拷入 AutoJS 项目根目录即可。

## 重新导出模型（可选）

```bash
pip install ultralytics ncnn pnnx==20260526   # pnnx 版本被官方钉死（新版有段错误 bug）
yolo export model=yolo26n.pt format=ncnn      # 产物在 yolo26n_ncnn_model/
# 将 model.ncnn.param / model.ncnn.bin 放入 yolo26/src/main/assets/ 即可
```

关键约定（C++ 解码依赖）：
- 导出为**静态 640×640** 输入（`inputshape=[1,3,640,640]`），letterbox 必须精确补到 640×640
- 单输出 `out0`，形状 `w=8400(锚点) × h=84(通道)`，通道 0-3 为已解码 xywh（640 坐标系），4.. 为已 sigmoid 类别分数
- 换不同类别数 nc 的模型：输出通道变为 4+nc，代码自动适配（`decode_decoded`）；若 nc 使得通道数 >100 会误判为 DFL 原始模式，需调整 `decode_output` 中阈值

## 构建 AAR（本机已配置好，一键）

```bash
# 依赖: JDK 17, Android SDK(platform 34 / build-tools 34 / NDK 26.3 / CMake 3.22), Gradle 8.9
gradle :yolo26:assembleRelease          # 或 ./gradlew :yolo26:assembleRelease
# 产物: yolo26/build/outputs/aar/yolo26-release.aar
```

## 工程结构

```
yolo26/
├── build.gradle.kts            # com.android.library, abiFilters: arm64-v8a/armeabi-v7a/x86_64
└── src/main/
    ├── assets/                 # yolo26n_ncnn_model.param/.bin (随 AAR 分发)
    ├── jniLibs/<abi>/libncnn.so
    ├── cpp/
    │   ├── CMakeLists.txt      # 链接官方 libncnn.so
    │   └── yolo26_jni.cpp      # 引擎: letterbox → 推理 → 解码 → NMS → JNI
    └── java/com/yolo26/ncnn/
        ├── Yolo26NcnnDetector.java   # 公共 API（纯 Java，实例式）
        └── Detection.java
```

## 已知限制

- CPU 推理（fp16 存储），无 Vulkan 依赖；采用官方动态库（OpenMP 构建，实测最快，113ms @ 640）。崩溃已通过模块 `useCache=false` 修复（根因是 AutoJS 按 dex 缓存加载器导致旧库被复用，与 OpenMP 无关）。若遇 OpenMP 兼容问题，备选方案为自编无 OpenMP 版（实测慢约 30-50%）
- 推理尺寸 = 模型导出尺寸（param/bin 配套），换分辨率需重新导出（`yolo export ... imgsz=xxx`）；`inputSize=0` 自动识别
- 未做 int8 量化（约 2-3 倍提速，需 ncnn 量化工具 + 校准集）
