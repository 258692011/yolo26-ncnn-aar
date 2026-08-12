package com.yolo26.ncnn;

/**
 * 单个检测结果，坐标已映射回原图（未裁剪）。
 * 字段可通过 getLabel()/getScore()/getX1()/... 访问（AutoJS 中也可用 .label/.score 等属性）。
 */
public class Detection {
    private final int label;
    private final float score;
    private final float x1;
    private final float y1;
    private final float x2;
    private final float y2;

    public Detection(int label, float score, float x1, float y1, float x2, float y2) {
        this.label = label;
        this.score = score;
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
    }

    public int getLabel() {
        return label;
    }

    public float getScore() {
        return score;
    }

    public float getX1() {
        return x1;
    }

    public float getY1() {
        return y1;
    }

    public float getX2() {
        return x2;
    }

    public float getY2() {
        return y2;
    }

    public float getWidth() {
        return x2 - x1;
    }

    public float getHeight() {
        return y2 - y1;
    }

    @Override
    public String toString() {
        return "Detection{label=" + label + ", score=" + score + ", box=(" + x1 + "," + y1 + "," + x2 + "," + y2 + ")}";
    }
}
