package com.tzutalin.dlib;

import android.os.Environment;

import java.io.File;

/**
 * Created by darrenl on 2016/4/22.
 */
public final class Constants {
    private Constants() {
        // Constants should be prive
    }

    /**
     * getFaceShapeModelPath
     * @return default face shape model path
     */
    public static String getFaceShapeModelPath() {
        File sdcard = Environment.getExternalStorageDirectory();
        String targetPath = sdcard.getAbsolutePath() + File.separator + "shape_predictor_68_face_landmarks.dat";
        return targetPath;
    }

    public static String getFrontalCascade() {
        File sdcard = Environment.getExternalStorageDirectory();
        String targetPath = sdcard.getAbsolutePath() + File.separator + "lbpcascade_frontalface.xml";
        return targetPath;
    }

    public static String getHaarCascade() {
        File sdcard = Environment.getExternalStorageDirectory();
        String targetPath = sdcard.getAbsolutePath() + File.separator + "haarcascade_lefteye_2splits.xml";
        return targetPath;
    }
}
