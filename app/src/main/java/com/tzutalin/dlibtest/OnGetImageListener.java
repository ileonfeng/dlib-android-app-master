/*
 * Copyright 2016 Tzutalin
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.tzutalin.dlibtest;

import android.content.Context;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.Rect;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Handler;
import android.os.Trace;
import android.util.Log;
import android.view.Display;
import android.view.WindowManager;

import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.PeopleDet;
import com.tzutalin.dlib.VisionDetRet;

import junit.framework.Assert;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point3;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import static org.opencv.calib3d.Calib3d.projectPoints;


/**
 * Class that takes in preview frames and converts the image to Bitmaps to process with dlib lib.
 */
public class OnGetImageListener implements OnImageAvailableListener {
    private static final boolean SAVE_PREVIEW_BITMAP = false;

    private static final int NUM_CLASSES = 1001;
    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 117;
    private static final String TAG = "OnGetImageListener";

    private double VECTOR_SIZE = 400.0;

    private int mScreenRotation = 270;

    private int mPreviewWdith = 0;
    private int mPreviewHeight = 0;
    private byte[][] mYUVBytes;
    private int[] mRGBBytes = null;
    private Bitmap mRGBframeBitmap = null;
    private Bitmap mCroppedBitmap = null;


    private File                   mCascadeFile;
    private File                   mCascadeFileEye;

    private boolean mIsComputing = false;
    private Handler mInferenceHandler;

    private Context mContext;
    private PeopleDet mPeopleDet;
    private TrasparentTitleView mTransparentTitleView;
    private FloatingCameraWindow mWindow;
    private Paint mFaceLandmardkPaint;

    final private int kEyePercentTop = 25;
    final private int kEyePercentSide = 13;
    final private int kEyePercentHeight = 30;
    final private int kEyePercentWidth = 35;

    public void initialize(
            final Context context,
            final AssetManager assetManager,
            final TrasparentTitleView scoreView,
            final Handler handler) {
        this.mContext = context;
        this.mTransparentTitleView = scoreView;
        this.mInferenceHandler = handler;
        mPeopleDet = new PeopleDet();
        mWindow = new FloatingCameraWindow(mContext);

        mFaceLandmardkPaint = new Paint();
        mFaceLandmardkPaint.setColor(Color.BLUE);
        mFaceLandmardkPaint.setStrokeWidth(2);
        mFaceLandmardkPaint.setStyle(Paint.Style.STROKE);
    }

    public void deInitialize() {
        synchronized (OnGetImageListener.this) {
            if (mPeopleDet != null) {
                mPeopleDet.deInit();
            }

            if (mWindow != null) {
                mWindow.release();
            }
        }
    }

    private void drawResizedBitmap(final Bitmap src, final Bitmap dst) {

        Display getOrient = ((WindowManager) mContext.getSystemService(Context.WINDOW_SERVICE)).getDefaultDisplay();
        int orientation = Configuration.ORIENTATION_UNDEFINED;
        Point point = new Point();
        getOrient.getSize(point);
        int screen_width = point.x;
        int screen_height = point.y;
        Log.d(TAG, String.format("screen size (%d,%d)", screen_width, screen_height));
        if (screen_width < screen_height) {
            orientation = Configuration.ORIENTATION_PORTRAIT;
            mScreenRotation = 270;
        } else {
            orientation = Configuration.ORIENTATION_LANDSCAPE;
            mScreenRotation = 0;
        }

        Assert.assertEquals(dst.getWidth(), dst.getHeight());
        final float minDim = Math.min(src.getWidth(), src.getHeight());

        final Matrix matrix = new Matrix();

        // We only want the center square out of the original rectangle.
        final float translateX = -Math.max(0, (src.getWidth() - minDim) / 2);
        final float translateY = -Math.max(0, (src.getHeight() - minDim) / 2);
        matrix.preTranslate(translateX, translateY);

        final float scaleFactor = dst.getHeight() / minDim;
        //matrix.postScale(-1,1);
        matrix.postScale(scaleFactor, scaleFactor);

        // Rotate around the center if necessary.
        if (mScreenRotation != 0) {
            matrix.postTranslate(-dst.getWidth() / 2.0f, -dst.getHeight() / 2.0f);
            matrix.postRotate(mScreenRotation);
            matrix.postTranslate(dst.getWidth() / 2.0f, dst.getHeight() / 2.0f);
        }


        final Canvas canvas = new Canvas(dst);
        //canvas.scale(-1,1);
        //matrix.preScale(-1,1);
        canvas.drawBitmap(src, matrix, null);
    }


    public MatOfPoint3f get_3d_model_points()
    {
        List<Point3> objectPointsList    = new ArrayList<Point3>(46);
        objectPointsList.add(new Point3(0.0f, 0.0f, 0.0f));
        objectPointsList.add(new Point3(0.0f, -421.4845f, -89.3745f));
        objectPointsList.add(new Point3(-247.603f, 122.219f, -237.55f));
        objectPointsList.add(new Point3(247.603f, 122.219f, -237.55f));
        objectPointsList.add(new Point3(-123.685f, -204.6945f, -120.4095f));
        objectPointsList.add(new Point3(123.685f, -204.6945f, -120.4095f));

        objectPointsList.add(new Point3(63.775f, -68.543f, -76.4535f));
        objectPointsList.add(new Point3(-63.775f, -68.543f, -76.4535f));

        objectPointsList.add(new Point3(0f, 151.374f, -103.856f));

        objectPointsList.add(new Point3(-372.819f, 177.768f, -487.865f));//0
        objectPointsList.add(new Point3(372.819f, 177.768f, -487.865f));//16

        objectPointsList.add(new Point3(-368.8095f, 80.0575f, -479.0175f));//1
        objectPointsList.add(new Point3(368.8095f, 80.0575f, -479.0175f));//15

        objectPointsList.add(new Point3(-353.2915f, -51.7435f, -443.356f));//2
        objectPointsList.add(new Point3(353.2915f, -51.7435f, -443.356f));//14

        objectPointsList.add(new Point3(-333.1235f, -156.7025f, -407.6445f));//3
        objectPointsList.add(new Point3(333.1235f, -156.7025f, -407.6445f));//13

        objectPointsList.add(new Point3(-305.707f, -229.4145f, -345.786));//4
        objectPointsList.add(new Point3(305.707f, -229.4145f, -345.786));//12

        objectPointsList.add(new Point3(-252.414f, -310.842f, -276.34f));//5
        objectPointsList.add(new Point3(252.414f, -310.842f, -276.34f));//11

        objectPointsList.add(new Point3(-175.322f, -369.1215f, -202.9945f));//6
        objectPointsList.add(new Point3(175.322f, -369.1215f, -202.9945f));//10

        objectPointsList.add(new Point3(-101.583f, -411.735f, -143.4995f));//7
        objectPointsList.add(new Point3(101.583f, -411.735f, -143.4995f));//9

        //******************************EYES**************************************

        objectPointsList.add(new Point3(-206.45f, 162.5685f, -168.746f));//37
        objectPointsList.add(new Point3(206.45f, 162.5685f, -168.746f));//43

        objectPointsList.add(new Point3(-145.069f, 162.436f, -168.746f));//38
        objectPointsList.add(new Point3(145.069f, 162.436f, -168.746f));//44

        objectPointsList.add(new Point3(-84.0115f, 132.2425f, -186.053f));//39
        objectPointsList.add(new Point3(84.0115f, 132.2425f, -186.053f));//42

        objectPointsList.add(new Point3(-135.093f, 110.194f, -176.4535f));//40
        objectPointsList.add(new Point3(135.093f, 110.194f, -176.4535f));//46

        objectPointsList.add(new Point3(-194.689f, 99.825f, -188.2525f));//41
        objectPointsList.add(new Point3(194.689f, 99.825f, -188.2525f));//47

        //******************************NOSE**************************************

        objectPointsList.add(new Point3(0f, 118.2505f, -80.8105f));//28
        objectPointsList.add(new Point3(0f, 52.0955f, -40.376f));//29
        objectPointsList.add(new Point3(-25.6955f, -80.0655f, -66.1125f));//32
        objectPointsList.add(new Point3(-1.043f, -83.0955f, -59.773f));//33
        objectPointsList.add(new Point3(25.6955f, -80.0655f, -66.1125f));//34

        //******************************LIPS**************************************

        objectPointsList.add(new Point3(2.782f, -162.207f, -55.933f));//51
        objectPointsList.add(new Point3(34.8215f, -152.1205f, -60.389f));//52
        objectPointsList.add(new Point3(-34.8215f, -152.1205f, -60.389f));//50
        objectPointsList.add(new Point3(43.516f, -228.234f, -72.8285f));//56
        objectPointsList.add(new Point3(-43.516f, -228.234f, -72.8285f));//58
        objectPointsList.add(new Point3(11.231f, -244.0305f, -71.3645f));//57


        MatOfPoint3f modelPoints = new MatOfPoint3f();
        modelPoints.fromList(objectPointsList);

        return modelPoints;

    }

    public MatOfPoint2f get_2d_image_points(ArrayList<Point> d, boolean isCenter)
    {
        List<org.opencv.core.Point> imagePointsList      = new ArrayList<org.opencv.core.Point>(46);
        float offsetx = 0;
        float offsety = 0;

        if (isCenter) {
            offsetx = mCroppedBitmap.getWidth() / 2 - d.get(30).x;
            offsety = mCroppedBitmap.getHeight() / 2 - d.get(30).y;
        }

        imagePointsList.add(new org.opencv.core.Point(d.get(30).x + offsetx, d.get(30).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(8).x + offsetx, d.get(8).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(36).x + offsetx, d.get(36).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(45).x + offsetx, d.get(45).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(48).x + offsetx, d.get(48).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(54).x + offsetx, d.get(54).y + offsety));

        imagePointsList.add(new org.opencv.core.Point(d.get(35).x + offsetx, d.get(35).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(31).x + offsetx, d.get(31).y + offsety));

        imagePointsList.add(new org.opencv.core.Point(d.get(27).x + offsetx, d.get(27).y + offsety));

        //************************FACE STROKE*************************************
        imagePointsList.add(new org.opencv.core.Point(d.get(0).x + offsetx, d.get(0).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(16).x + offsetx, d.get(16).y + offsety));

        imagePointsList.add(new org.opencv.core.Point(d.get(1).x + offsetx, d.get(1).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(15).x + offsetx, d.get(15).y + offsety));

        imagePointsList.add(new org.opencv.core.Point(d.get(2).x + offsetx, d.get(2).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(14).x + offsetx, d.get(14).y + offsety));

        imagePointsList.add(new org.opencv.core.Point(d.get(3).x + offsetx, d.get(3).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(13).x + offsetx, d.get(13).y + offsety));

        imagePointsList.add(new org.opencv.core.Point(d.get(4).x + offsetx, d.get(4).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(12).x + offsetx, d.get(12).y + offsety));

        imagePointsList.add(new org.opencv.core.Point(d.get(5).x + offsetx, d.get(5).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(11).x + offsetx, d.get(11).y + offsety));

        imagePointsList.add(new org.opencv.core.Point(d.get(6).x + offsetx, d.get(6).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(10).x + offsetx, d.get(10).y + offsety));

        imagePointsList.add(new org.opencv.core.Point(d.get(7).x + offsetx, d.get(7).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(9).x + offsetx, d.get(9).y + offsety));

        //*****************************.y + offsetyES*****************************************

        imagePointsList.add(new org.opencv.core.Point(d.get(37).x + offsetx, d.get(37).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(43).x + offsetx, d.get(43).y + offsety));

        imagePointsList.add(new org.opencv.core.Point(d.get(38).x + offsetx, d.get(38).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(44).x + offsetx, d.get(44).y + offsety));

        imagePointsList.add(new org.opencv.core.Point(d.get(39).x + offsetx, d.get(39).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(42).x + offsetx, d.get(42).y + offsety));

        imagePointsList.add(new org.opencv.core.Point(d.get(40).x + offsetx, d.get(40).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(46).x + offsetx, d.get(46).y + offsety));

        imagePointsList.add(new org.opencv.core.Point(d.get(41).x + offsetx, d.get(41).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(47).x + offsetx, d.get(47).y + offsety));

        //*****************************NOSE*****************************************
        imagePointsList.add(new org.opencv.core.Point(d.get(28).x + offsetx, d.get(28).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(29).x + offsetx, d.get(29).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(32).x + offsetx, d.get(32).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(33).x + offsetx, d.get(33).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(34).x + offsetx, d.get(34).y + offsety));

        //***************************LIPS******************************************
        imagePointsList.add(new org.opencv.core.Point(d.get(51).x + offsetx, d.get(51).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(52).x + offsetx, d.get(52).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(50).x + offsetx, d.get(50).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(56).x + offsetx, d.get(56).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(58).x + offsetx, d.get(58).y + offsety));
        imagePointsList.add(new org.opencv.core.Point(d.get(57).x + offsetx, d.get(57).y + offsety));

        MatOfPoint2f modelPoints = new MatOfPoint2f();
        modelPoints.fromList(imagePointsList);
        return modelPoints;
    }

    Mat get_camera_matrix(float focal_length, Point center)
    {
        Mat camera_matrix =  Mat.eye(3, 3, CvType.CV_32F);

        camera_matrix.put(0,0,focal_length);
        camera_matrix.put(0,1,0.0);
        camera_matrix.put(0,2,center.x);

        camera_matrix.put(1,0,0.0);
        camera_matrix.put(1,1,focal_length);
        camera_matrix.put(1,2,center.y);

        camera_matrix.put(2,0,0.0);
        camera_matrix.put(2,1,0.0);
        camera_matrix.put(2,2,1.0);
        return camera_matrix;
    }

    public void onlyHeadPoseLandmarks(Canvas canvas, ArrayList<Point> d){
        canvas.drawCircle(d.get(30).x, d.get(30).y, 1, mFaceLandmardkPaint);
        canvas.drawCircle(d.get(8).x, d.get(8).y, 1, mFaceLandmardkPaint);
        canvas.drawCircle(d.get(36).x, d.get(36).y, 1, mFaceLandmardkPaint);
        canvas.drawCircle(d.get(45).x, d.get(45).y, 1, mFaceLandmardkPaint);
        canvas.drawCircle(d.get(48).x, d.get(48).y, 1, mFaceLandmardkPaint);
        canvas.drawCircle(d.get(54).x, d.get(54).y, 1, mFaceLandmardkPaint);
    }

    public enum Direction { VERTICAL, HORIZONTAL };

    public static Bitmap flip(Bitmap src, Direction type) {
        Matrix matrix = new Matrix();

        if(type == Direction.VERTICAL) {
            matrix.preScale(1.0f, -1.0f);
        }
        else if(type == Direction.HORIZONTAL) {
            matrix.preScale(-1.0f, 1.0f);
        } else {
            return src;
        }

        return Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), matrix, true);
    }

    @Override
    public void onImageAvailable(final ImageReader reader) {
        Image image = null;
        try {
            image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            // No mutex needed as this method is not reentrant.
            if (mIsComputing) {
                image.close();
                return;
            }
            mIsComputing = true;

            Trace.beginSection("imageAvailable");

            final Plane[] planes = image.getPlanes();

            // Initialize the storage bitmaps once when the resolution is known.
            if (mPreviewWdith != image.getWidth() || mPreviewHeight != image.getHeight()) {
                mPreviewWdith = image.getWidth();
                mPreviewHeight = image.getHeight();

                Log.d(TAG, String.format("Initializing at size %dx%d", mPreviewWdith, mPreviewHeight));
                mRGBBytes = new int[mPreviewWdith * mPreviewHeight];
                mRGBframeBitmap = Bitmap.createBitmap(mPreviewWdith, mPreviewHeight, Config.ARGB_8888);
                mCroppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Config.ARGB_8888);

                mYUVBytes = new byte[planes.length][];
                for (int i = 0; i < planes.length; ++i) {
                    mYUVBytes[i] = new byte[planes[i].getBuffer().capacity()];
                }
            }

            for (int i = 0; i < planes.length; ++i) {
                planes[i].getBuffer().get(mYUVBytes[i]);
            }

            final int yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();
            ImageUtils.convertYUV420ToARGB8888(
                    mYUVBytes[0],
                    mYUVBytes[1],
                    mYUVBytes[2],
                    mRGBBytes,
                    mPreviewWdith,
                    mPreviewHeight,
                    yRowStride,
                    uvRowStride,
                    uvPixelStride,
                    false);

            image.close();
        } catch (final Exception e) {
            if (image != null) {
                image.close();
            }
            Log.e(TAG, "Exception!", e);
            Trace.endSection();
            return;
        }

        mRGBframeBitmap.setPixels(mRGBBytes, 0, mPreviewWdith, 0, 0, mPreviewWdith, mPreviewHeight);
        drawResizedBitmap(mRGBframeBitmap, mCroppedBitmap);
/*
        Matrix m = new Matrix();
        m.preScale(-1, 1);
        Bitmap dst = Bitmap.createBitmap(mCroppedBitmap, 0, 0, mCroppedBitmap.getWidth(), mCroppedBitmap.getHeight(), m, false);
        dst.setDensity(DisplayMetrics.DENSITY_DEFAULT);
        mCroppedBitmap = dst;*/

        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(mCroppedBitmap);
        }

        mInferenceHandler.post(
                new Runnable() {
                    @Override
                    public void run() {
                        final String targetPath = Constants.getFaceShapeModelPath();
                        if (!new File(targetPath).exists()) {
                            mTransparentTitleView.setText("Copying landmark model to " + targetPath);
                            FileUtils.copyFileFromRawToOthers(mContext, R.raw.shape_predictor_68_face_landmarks, targetPath);
                        }

                        //long startTime = System.currentTimeMillis();
                        List<VisionDetRet> results;
                        synchronized (OnGetImageListener.this) {
                            results = mPeopleDet.detBitmapFace(mCroppedBitmap, targetPath);
                        }
                        //long endTime = System.currentTimeMillis();
                        //mTransparentTitleView.setText("Time cost: " + String.valueOf((endTime - startTime) / 1000f) + " sec");
                        // Draw on bitmap
                        if (results != null) {
                            for (final VisionDetRet ret : results) {
                                float resizeRatio = 1.0f;

                                Canvas canvas = new Canvas(mCroppedBitmap);

                                org.opencv.core.Rect face = new org.opencv.core.Rect(ret.getLeft(),ret.getTop(),ret.getRight() - ret.getLeft(),ret.getBottom() - ret.getTop());
                                Mat gray = new Mat (mCroppedBitmap.getWidth(), mCroppedBitmap.getHeight(), CvType.CV_8UC1);
                                Utils.bitmapToMat(mCroppedBitmap, gray);
                                Imgproc.cvtColor(gray, gray, Imgproc.COLOR_RGB2GRAY);
                                Mat roi = gray.submat(face);


                                // Draw landmark
                                ArrayList<Point> landmarks = ret.getFaceLandmarks();
                                MatOfPoint3f model_points = get_3d_model_points();
                                MatOfPoint2f image_points  = get_2d_image_points(landmarks, true);
                                int focal_length = mCroppedBitmap.getWidth();

                                Mat camera_matrix = get_camera_matrix(focal_length, new Point(mCroppedBitmap.getHeight()/2, mCroppedBitmap.getWidth()/2));
                                MatOfDouble dist_coeffs = new MatOfDouble(Mat.zeros(4,1, CvType.CV_64FC1));

                                Mat rvec = new Mat();
                                Mat tvec = new Mat();

                                Calib3d.solvePnP(model_points, image_points, camera_matrix, new MatOfDouble(), rvec, tvec);
                                Mat Rmat = Mat.eye(3, 3, CvType.CV_32F);
                                Mat Rmat2 = Mat.eye(3, 3, CvType.CV_32F);

                                Calib3d.Rodrigues(rvec, Rmat);
                                Mat ans = Mat.eye(3, 1, CvType.CV_32F);
                                Rmat.convertTo(Rmat2,-1,-1);
                                Core.gemm(Rmat2,tvec, 1, Mat.eye(3, 3, CvType.CV_32F),0,ans, 0);
                                List<Point3> objectPointsList    = new ArrayList<Point3>(1);
                                objectPointsList.add(new Point3(0,0,VECTOR_SIZE));
                                //objectPointsList.add(new Point3(0,500,0));
                                MatOfPoint3f nose = new MatOfPoint3f();
                                MatOfPoint3f c3d = new MatOfPoint3f();
                                nose.fromList(objectPointsList);
                                List<Point3> objectPointsList2    = new ArrayList<Point3>(1);
                                double t = ans.get(2,0)[0] / VECTOR_SIZE;
                                objectPointsList2.add(new Point3((ans.get(0,0)[0]) / t, (ans.get(1,0)[0]) / t, (ans.get(2,0)[0]) / t));
                                c3d.fromList(objectPointsList2);
                                MatOfPoint2f nose2 = new MatOfPoint2f();
                                MatOfPoint2f camera = new MatOfPoint2f();

                                projectPoints(nose,rvec,tvec,camera_matrix,new MatOfDouble(),nose2);
                                projectPoints(c3d, rvec, tvec, camera_matrix, new MatOfDouble(), camera);

                                for (Point point : landmarks) {
                                    int pointX = (int) (point.x * resizeRatio);
                                    int pointY = (int) (point.y * resizeRatio);

                                    canvas.drawCircle(pointX, pointY, 1, mFaceLandmardkPaint);
                                }
                                //onlyHeadPoseLandmarks(canvas,landmarks);


                                Mat im = new Mat();
                                Utils.bitmapToMat(mCroppedBitmap, im);
                                org.opencv.core.Point looking_point = nose2.toList().get(0);
                                org.opencv.core.Point nose_point = image_points.toList().get(0);
                                org.opencv.core.Point3 camera_point = c3d.toList().get(0);
                                int cam_x = mCroppedBitmap.getWidth()/2;
                                int cam_y = mCroppedBitmap.getHeight()/2;
                                Double dist_nose_look_xz = Math.sqrt(Math.pow(0-0,2)+Math.pow(0-VECTOR_SIZE,2));
                                Double dist_nose_camera_xz = Math.sqrt(Math.pow(0-camera_point.x,2)+Math.pow(0-camera_point.z,2));
                                Double dist_look_camera_xz = Math.sqrt(Math.pow(0-camera_point.x,2)+Math.pow(VECTOR_SIZE-camera_point.z,2));

                                Double dist_nose_look_yz = Math.sqrt(Math.pow(0-0,2)+Math.pow(0-VECTOR_SIZE,2));
                                Double dist_nose_camera_yz = Math.sqrt(Math.pow(0-camera_point.y,2)+Math.pow(0-camera_point.z,2));
                                Double dist_look_camera_yz = Math.sqrt(Math.pow(0-camera_point.y,2)+Math.pow(VECTOR_SIZE-camera_point.z,2));

                                Double head_angle_xz = Math.acos((Math.pow(dist_look_camera_xz,2)-Math.pow(dist_nose_look_xz,2)-Math.pow(dist_nose_camera_xz,2))/(-2*dist_nose_look_xz*dist_nose_camera_xz));
                                Double head_angle_yz = Math.acos((Math.pow(dist_look_camera_yz,2)-Math.pow(dist_nose_look_yz,2)-Math.pow(dist_nose_camera_yz,2))/(-2*dist_nose_look_yz*dist_nose_camera_yz));

                                //Double head_angle_y = Math.asin(Math.abs(looking_point.y - nose_point.y)/Math.sqrt(Math.pow(looking_point.x-nose_point.x,2)+Math.pow(looking_point.y-nose_point.y,2)));
                                //Double head_angle_x = Math.asin(Math.abs(looking_point.x - nose_point.x)/Math.sqrt(Math.pow(looking_point.x-nose_point.x,2)+Math.pow(looking_point.z-nose_point.z,2)));
                                mTransparentTitleView.setText("head yz: "+String.format("%1$,.2f",head_angle_yz*180/Math.PI)+" head xz: "+String.format("%1$,.2f",head_angle_xz*180/Math.PI)+
                                                            " ("+String.format("%1$,.2f",camera_point.x)+", "+String.format("%1$,.2f",camera_point.y)+", "+String.format("%1$,.2f",camera_point.z)+")");
                                Imgproc.line(im,nose_point,looking_point,new Scalar(255,0,0),2 );
                                Imgproc.line(im,nose_point,camera.toList().get(0),new Scalar(0,0,255),2 );
                                Utils.matToBitmap(im,mCroppedBitmap);

                            }
                        }

                        //mCroppedBitmap = flip(mCroppedBitmap,Direction.HORIZONTAL);
                        mWindow.setRGBBitmap(mCroppedBitmap);
                        mIsComputing = false;
                    }
                });

        Trace.endSection();
    }

}
