package org.opencv.android;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.DialogInterface.OnClickListener;
import android.util.Log;

/**
 * Basic implementation of LoaderCallbackInterface.
 */
public abstract class BaseLoaderCallback implements LoaderCallbackInterface {

    public BaseLoaderCallback(Context AppContext) {
        mAppContext = AppContext;
    }

    public void onManagerConnected(int status) {

    }

    public void onPackageInstall(final int operation, final InstallCallbackInterface callback)
    {
        switch (operation)
        {
            case InstallCallbackInterface.NEW_INSTALLATION:
            {
                AlertDialog InstallMessage = new AlertDialog.Builder(mAppContext).create();
                InstallMessage.setTitle("Package not found");
                InstallMessage.setMessage(callback.getPackageName() + " package was not found! Try to install it?");
                InstallMessage.setCancelable(false); // This blocks the 'BACK' button
                InstallMessage.setButton(AlertDialog.BUTTON_POSITIVE, "Yes", new OnClickListener()
                {
                    public void onClick(DialogInterface dialog, int which)
                    {
                        callback.install();
                    }
                });

                InstallMessage.setButton(AlertDialog.BUTTON_NEGATIVE, "No", new OnClickListener() {

                    public void onClick(DialogInterface dialog, int which)
                    {
                        callback.cancel();
                    }
                });

                InstallMessage.show();
            } break;
            case InstallCallbackInterface.INSTALLATION_PROGRESS:
            {
                AlertDialog WaitMessage = new AlertDialog.Builder(mAppContext).create();
                WaitMessage.setTitle("OpenCV is not ready");
                WaitMessage.setMessage("Installation is in progress. Wait or exit?");
                WaitMessage.setCancelable(false); // This blocks the 'BACK' button
                WaitMessage.setButton(AlertDialog.BUTTON_POSITIVE, "Wait", new OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        callback.wait_install();
                    }
                });
                WaitMessage.setButton(AlertDialog.BUTTON_NEGATIVE, "Exit", new OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        callback.cancel();
                    }
                });

                WaitMessage.show();
            } break;
        }
    }

    void finish()
    {
        ((Activity) mAppContext).finish();
    }

    protected Context mAppContext;
    private final static String TAG = "OpenCVLoader/BaseLoaderCallback";
}
