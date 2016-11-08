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

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.content.res.Resources;
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
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Display;
import android.view.WindowManager;

import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.PeopleDet;
import com.tzutalin.dlib.VisionDetRet;

import junit.framework.Assert;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point3;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import org.opencv.calib3d.Calib3d;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import static android.R.attr.type;
import static android.R.attr.width;
import static org.opencv.calib3d.Calib3d.projectPoints;
import static org.opencv.calib3d.Calib3d.solvePnP;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.TM_SQDIFF;
import static org.opencv.imgproc.Imgproc.cvtColor;


/**
 * Class that takes in preview frames and converts the image to Bitmaps to process with dlib lib.
 */
public class OnGetImageListener implements OnImageAvailableListener {
    private static final boolean SAVE_PREVIEW_BITMAP = false;

    private static final int NUM_CLASSES = 1001;
    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 117;
    private static final String TAG = "OnGetImageListener";

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
        List<Point3> objectPointsList    = new ArrayList<Point3>(6);
        objectPointsList.add(new Point3(0.0f, 0.0f, 0.0f));
        objectPointsList.add(new Point3(0.0f, -330.0f, -65.0f));
        objectPointsList.add(new Point3(-225.0f, 170.0f, -135.0f));
        objectPointsList.add(new Point3(225.0f, 170.0f, -135.0f));
        objectPointsList.add(new Point3(-150.0f, -150.0f, -125.0f));
        objectPointsList.add(new Point3(150.0f, -150.0f, -125.0f));

        MatOfPoint3f modelPoints = new MatOfPoint3f();
        modelPoints.fromList(objectPointsList);

        return modelPoints;

    }

    public MatOfPoint2f get_2d_image_points(ArrayList<Point> d)
    {
        List<org.opencv.core.Point> imagePointsList      = new ArrayList<org.opencv.core.Point>(6);
        imagePointsList.add(new org.opencv.core.Point(d.get(30).x, d.get(30).y));
        imagePointsList.add(new org.opencv.core.Point(d.get(8).x, d.get(8).y));
        imagePointsList.add(new org.opencv.core.Point(d.get(36).x, d.get(36).y));
        imagePointsList.add(new org.opencv.core.Point(d.get(45).x, d.get(45).y));
        imagePointsList.add(new org.opencv.core.Point(d.get(48).x, d.get(48).y));
        imagePointsList.add(new org.opencv.core.Point(d.get(54).x, d.get(54).y));

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

                                // Draw landmark
                                ArrayList<Point> landmarks = ret.getFaceLandmarks();
                                MatOfPoint3f model_points = get_3d_model_points();
                                MatOfPoint2f image_points  = get_2d_image_points(landmarks);
                                int focal_length = mCroppedBitmap.getHeight();

                                Mat camera_matrix = get_camera_matrix(focal_length, new Point(mCroppedBitmap.getWidth()/2, mCroppedBitmap.getHeight()/2));
                                MatOfDouble dist_coeffs = new MatOfDouble(Mat.zeros(4,1, CvType.CV_64FC1));

                                Mat rvec = new Mat();
                                Mat tvec = new Mat();

                                Calib3d.solvePnP(model_points, image_points, camera_matrix, new MatOfDouble(), rvec, tvec);

                                List<Point3> objectPointsList    = new ArrayList<Point3>(1);
                                objectPointsList.add(new Point3(0,0,1000.0));
                                MatOfPoint3f nose = new MatOfPoint3f();
                                nose.fromList(objectPointsList);

                                MatOfPoint2f nose2 = new MatOfPoint2f();

                                projectPoints(nose,rvec,tvec,camera_matrix,new MatOfDouble(),nose2);

                                for (Point point : landmarks) {
                                    int pointX = (int) (point.x * resizeRatio);
                                    int pointY = (int) (point.y * resizeRatio);

                                    canvas.drawCircle(pointX, pointY, 1, mFaceLandmardkPaint);
                                }

                                //onlyHeadPoseLandmarks(canvas,landmarks);


                                Mat im = new Mat();
                                Utils.bitmapToMat(mCroppedBitmap, im);
                                Imgproc.line(im,image_points.toList().get(0),nose2.toList().get(0),new Scalar(255,0,0),2 );

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
