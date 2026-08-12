plugins {
    id("com.android.library")
}

android {
    namespace = "com.yolo26.ncnn"
    compileSdk = 34

    defaultConfig {
        minSdk = 24
        targetSdk = 34

        // 钉住已安装的 NDK，避免 AGP 自动下载其默认版本 (26.1，约 1GB)
        ndkVersion = "26.3.11579264"

        ndk {
            // 真机主流 arm64-v8a；armeabi-v7a 覆盖老设备；x86_64/x86 供模拟器
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64", "x86")
        }

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17 -O2"
                arguments += "-DANDROID_STL=c++_static"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"  // 钉住已安装版本，避免 AGP 额外下载
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
