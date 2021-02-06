package com.facefinder.jordigarnacho.facefinder;

// Importations des librairies
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.net.Uri;
import android.os.Bundle;
import android.os.StrictMode;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import java.io.IOException;
import java.text.SimpleDateFormat;
import android.os.Environment;
import com.google.android.gms.appindexing.Action;
import com.google.android.gms.appindexing.AppIndex;
import com.google.android.gms.appindexing.Thing;
import com.google.android.gms.common.api.GoogleApiClient;
import java.io.File;
import java.util.Date;
import android.content.Intent;
import android.provider.MediaStore;
import android.widget.TextView;
import android.widget.Toast;
import java.io.*;
import java.net.*;

// Activité principale (et unique)
public class MainActivity extends AppCompatActivity {

    // Déclaration des variables globales de l'activité

    // Variable pour l'appareil photo
    private Camera mCamera = null;

    // Variable pour la gestion de l'affichage du
    // flux vidéo de l'appareil photo
    private CameraView mCameraView = null;

    // Variable pour le stockage des fichiers
    // images (JPEG)
    public File imageFile = null;

    // Variables pour la décomposition de
    // l'adresse du serveur
    private String serverIP1 = "192";
    private String serverIP2 = "168";
    private String serverIP3 = "118";
    private String serverIP4 = "102";

    // Variable pour le port du serveur
    private int serverPort = 8000;

    /**
     * ATTENTION: This was auto-generated to implement the App Indexing API.
     * See https://g.co/AppIndexing/AndroidStudio for more information.
     */
    private GoogleApiClient client;

    @Override

    // Fonction lancée lors du lancement de l'application
    // et de la première activité
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Execution de la fonction gérant le menu
        // principal de l'application
        mainPageActions();

        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        client = new GoogleApiClient.Builder(this).addApi(AppIndex.API).build();
    }

    // Fonction permettant d'écouter un
    // évènement provenant de l'appui sur
    // un bouton du menu ajouté dans l'Action Bar
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()){

            // Cas d'un click sur le logo
            // de l'application dans l'Action Bar
            case R.id.menuHomeButton:

                // Execution de la fonction gérant le menu
                // principal de l'application
                mainPageActions();
                return true;

            // Cas d'un click sur Find My Face
            // de l'application dans l'Action Bar
            case R.id.menuFindMyFaceButton:

                // Execution de la fonction gérant
                // l'interface Find My Face
                findMyFaceActions();
                return true;

            // Cas d'un click sur Insert My Face
            // de l'application dans l'Action Bar
            case R.id.menuInsertMyFaceButton:

                // Execution de la fonction gérant
                // l'interface Insert My Face
                insertMyFaceActions();
                return true;

            // Cas d'un click sur Settings
            // de l'application dans l'Action Bar
            case R.id.menuSettingsButton:

                // Execution de la fonction gérant
                // l'interface Settings
                settingsActions();
                return true;

            // Cas d'un click sur About
            // de l'application dans l'Action Bar
            case R.id.menuAboutButton:

                // Execution de la fonction gérant
                // l'interface Actions
                aboutActions();
        }
        return super.onOptionsItemSelected(item);
    }

    // Fonction gérant l'interface du menu
    // principal
    public void mainPageActions(){

        // Redirection vers l'interface graphique activity_main
        setContentView(R.layout.activity_main);

        // Déclaration des boutons du menu
        Button findFaceButton = (Button) findViewById(R.id.FindFaceButton);
        Button insertMyFaceButton = (Button) findViewById(R.id.InsertFaceButton);
        Button settingsButton = (Button) findViewById(R.id.SettingsButton);
        Button aboutButton = (Button) findViewById(R.id.AboutButton);

        // Actions lors d'un click sur le
        // bouton Find My Face
        findFaceButton.setOnClickListener(new OnClickListener() {
            public void onClick(View v) {

                // Execution de la fonction gérant
                // l'interface Find My Face
                findMyFaceActions();
            }
        });

        // Actions lors d'un click sur le
        // bouton Insert My Face
        insertMyFaceButton.setOnClickListener(new OnClickListener() {
            public void onClick(View v) {

                // Execution de la fonction gérant
                // l'interface Insert My Face
                insertMyFaceActions();
            }
        });

        // Actions lors d'un click sur le
        // bouton Settings
        settingsButton.setOnClickListener(new OnClickListener() {
            public void onClick(View v) {

                // Execution de la fonction gérant
                // l'interface Settings
                settingsActions();
            };
        });

        // Actions lors d'un click sur le
        // bouton About
        aboutButton.setOnClickListener(new OnClickListener() {
            public void onClick(View v) {

                // Execution de la fonction gérant
                // l'interface About
                aboutActions();
            }
        });
    }

    // Fonction gérant l'interface Find My Face
    public void findMyFaceActions() {

        // Vérification que l'appareil possède une caméra.
        // Si l'appareil n'a pas de caméra, on ne lance pas le layout 'insert_my_face'
        // et on informe l'utilisateur.

        // Essai d'accès à la caméra de l'appareil.
        if (testPresenceCamera() != true)
        {
            Toast.makeText(MainActivity.super.getApplicationContext(), "Votre appareil n'a pas de caméra", Toast.LENGTH_SHORT).show();
            mainPageActions();
        }
        else {
            try {
                mCamera = Camera.open();//you can use open(int) to use different cameras
            }
            catch (Exception e) {
                e.printStackTrace();
            }
            if (testPresenceCamera() != true || mCamera == null) {
                // Message d'information.
                Toast.makeText(MainActivity.super.getApplicationContext(), "Votre appareil n'a pas de caméra", Toast.LENGTH_SHORT).show();
                mainPageActions();
            }

            // Si l'appareil possède de caméra, on lance le layout 'insert_my_face'
            // afin de capturer une photo.
            else {


                imageFile = null;

                // Lancement du layout 'insert_my_face'
                setContentView(R.layout.find_my_face);

                final ImageButton TakePictureButton = (ImageButton) findViewById(R.id.TakePictureButton);
                final ImageButton OKButton = (ImageButton) findViewById(R.id.TakeOkPictureButton);
                final ImageButton AgainButton = (ImageButton) findViewById(R.id.TakeAgainPictureButton);
                TakePictureButton.setVisibility(View.VISIBLE);
                OKButton.setVisibility(View.INVISIBLE);
                AgainButton.setVisibility(View.INVISIBLE);
                TakePictureButton.setEnabled(true);
                OKButton.setEnabled(false);
                AgainButton.setEnabled(false);

                // On créer une surface d'affichage du flux de la caméra.
                mCameraView = new CameraView(MainActivity.super.getApplicationContext(), mCamera);
                final FrameLayout camera_view = (FrameLayout) findViewById(R.id.camera_view);
                camera_view.addView(mCameraView);


                // Prise d'une photo.
                TakePictureButton.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {

                        // Enregistrement de la photo au format JPEG.

                        mCamera.takePicture(null, null, mPicture);

                        if (imageFile != null) {
                            Toast.makeText(MainActivity.super.getApplicationContext(), "La photo a été prise", Toast.LENGTH_SHORT).show();
                        }

                        TakePictureButton.setVisibility(View.INVISIBLE);
                        OKButton.setVisibility(View.VISIBLE);
                        AgainButton.setVisibility(View.VISIBLE);
                        TakePictureButton.setEnabled(false);
                        OKButton.setEnabled(true);
                        AgainButton.setEnabled(true);
                    }
                });

                // La photo ne convient pas
                AgainButton.setOnClickListener(new View.OnClickListener() {
                    public void onClick(View v) {

                        mCamera.startPreview();

                        TakePictureButton.setVisibility(View.VISIBLE);
                        OKButton.setVisibility(View.INVISIBLE);
                        AgainButton.setVisibility(View.INVISIBLE);
                        TakePictureButton.setEnabled(true);
                        OKButton.setEnabled(false);
                        AgainButton.setEnabled(false);
                    }
                });

                // La photo convient
                OKButton.setOnClickListener(new View.OnClickListener() {
                    public void onClick(View v) {
                        setContentView(R.layout.result_find_face);
                        String myName = "My User";
                        try {
                            myName = receiveTextServer();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                        TextView userFindField = (TextView) findViewById(R.id.userFindField);
                        userFindField.setText(myName);
                    }
                });
            }
        }
    }

    // Fonction gérant l'interface Insert My Face
    public void insertMyFaceActions() {

        // Vérification que l'appareil possède une caméra.
        // Si l'appareil n'a pas de caméra, on ne lance pas le layout 'insert_my_face'
        // et on informe l'utilisateur.

        // Essai d'accès à la caméra de l'appareil.
        if (testPresenceCamera() != true)
        {
            Toast.makeText(MainActivity.super.getApplicationContext(), "Votre appareil n'a pas de caméra", Toast.LENGTH_SHORT).show();
            mainPageActions();
        }
        else {
            try {
                mCamera = Camera.open();//you can use open(int) to use different cameras
            } catch (Exception e) {
                e.printStackTrace();
            }
            if (testPresenceCamera() != true || mCamera == null) {
                // Message d'information.
                Toast.makeText(MainActivity.super.getApplicationContext(), "Votre appareil n'a pas de caméra", Toast.LENGTH_SHORT).show();
                mainPageActions();
            }
            else {

                imageFile = null;
                setContentView(R.layout.insert_new_face);
                // Prise de la photo.

                if (mCamera != null) {
                    mCameraView = new CameraView(MainActivity.super.getApplicationContext(), mCamera);
                    FrameLayout camera_view = (FrameLayout) findViewById(R.id.camera_view);
                    camera_view.addView(mCameraView);//add the SurfaceView to the layout


                    Button SendNewUserButton = (Button) findViewById(R.id.SendNewUserButton);
                    ImageButton TakePictureButton2 = (ImageButton) findViewById(R.id.TakePictureButton2);
                    ImageButton TakeAgainPictureButton2 = (ImageButton) findViewById(R.id.TakeAgainPictureButton2);

                    TakePictureButton2.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View v) {

                            // Enregistrement de la photo au format JPEG.
                            mCamera.takePicture(null, null, mPicture);
                            if (imageFile != null) {
                                Toast.makeText(MainActivity.super.getApplicationContext(), "La photo a été prise", Toast.LENGTH_SHORT).show();
                            }
                        }
                    });
                    TakeAgainPictureButton2.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View v) {
                            mCamera.startPreview();
                        }
                    });

                    SendNewUserButton.setOnClickListener(new View.OnClickListener() {
                       @Override
                        public void onClick(View v) {

                            if (imageFile == null) {
                                Toast.makeText(MainActivity.super.getApplicationContext(), "Veuillez prendre un photo", Toast.LENGTH_SHORT).show();
                            }
                            else
                            {
                                String newUserFirstName = ((EditText) findViewById (R.id.newFaceFirstname)).getText().toString();
                                String newUserLastName = ((EditText) findViewById (R.id.newFaceLastname)).getText().toString();
                                File newUserPhoto = imageFile;

                                Toast.makeText(MainActivity.super.getApplicationContext(), "Envoi en cours...", Toast.LENGTH_SHORT).show();
                            }
                        }
                    });


                }
            }
        }
    }

    // Fonction gérant l'interface Settings
    public void settingsActions(){

        // Lancement du layout 'settings'
        setContentView(R.layout.settings);

        // Initialisation des champs de l'adresse IP
        // et du port du serveur avec les derniers paramètres
        // sauvegardés
        EditText ipServer1Field  = (EditText) findViewById(R.id.ipServer1);
        ipServer1Field.setText(serverIP1);
        EditText ipServer2Field  = (EditText) findViewById(R.id.ipServer2);
        ipServer2Field.setText(serverIP2);
        EditText ipServer3Field  = (EditText) findViewById(R.id.ipServer3);
        ipServer3Field.setText(serverIP3);
        EditText ipServer4Field  = (EditText) findViewById(R.id.ipServer4);
        ipServer4Field.setText(serverIP4.toString());
        EditText portServerField  = (EditText) findViewById(R.id.portServer);
        portServerField.setText(String.valueOf(serverPort));

        // Si click sur le bouton de validation
        // on sauvegarde les nouveaux paramètres
        Button validSettingsButton  = (Button) findViewById(R.id.validSettingsButton);
        validSettingsButton.setOnClickListener(new OnClickListener() {
            public void onClick(View v) {

                serverIP1 = ((EditText) findViewById (R.id.ipServer1)).getText().toString();
                serverIP2 = ((EditText) findViewById (R.id.ipServer2)).getText().toString();
                serverIP3 = ((EditText) findViewById (R.id.ipServer3)).getText().toString();
                serverIP4 = ((EditText) findViewById (R.id.ipServer4)).getText().toString();
                EditText portServerField  = (EditText) findViewById(R.id.portServer);
                serverPort = Integer.parseInt(portServerField.getText().toString());
                Toast.makeText(MainActivity.super.getApplicationContext(), "Paramètres sauvegardés", Toast.LENGTH_SHORT).show();
            }
        });
    }

    // Fonction gérant l'interface About
    public void aboutActions(){
        setContentView(R.layout.about);
    }

    // Fonction de test de présence
    // d'un appareil photo sur l'appareil.
    public boolean testPresenceCamera() {
        PackageManager packageManager = getPackageManager();
        return packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA);
    }


    // Fonction d'ajout du des boutons cliquables
    // dans l'Action Bar (menu).
    public boolean onCreateOptionsMenu(Menu myMenu) {
        getMenuInflater().inflate(R.menu.menu, myMenu);
        return true;
    }

    // Fonction permettant de recevoir un texte
    // provenant du serveur
    private String receiveTextServer() throws IOException {

        // Permission pour la création d'un connection via un socket
        StrictMode.ThreadPolicy policy = new StrictMode.ThreadPolicy.Builder().permitAll().build();
        StrictMode.setThreadPolicy(policy);

        // Récupération de l'adresse IP du serveur
        String myHost = serverIP1.toString()+"."+serverIP2.toString()+"."+serverIP3.toString()+"."+serverIP4.toString();

        // Déclaration et initialisation de la chaine
        // de caractère permettant de récupérer le message du serveur
        String inputLine = "";

        // Création de la socket et connection
        Socket mySocket = new Socket(myHost, serverPort);

        // Si la connection est réussi
        if (mySocket.isConnected() == true)
        {
            // Message d'information
            Toast.makeText(MainActivity.super.getApplicationContext(), "Connexion au serveur réussi", Toast.LENGTH_SHORT).show();

            // Déclaration du buffer de sortie du socket
            OutputStream outstream = mySocket.getOutputStream();

            // Déclaration de la variable permettant d'écrire
            // dans le buffer de sortie de la socket
            PrintWriter out = new PrintWriter(outstream, true);

            // Écriture de la valeur "2" dans le buffer de sortie afin d'indiquer
            // que l'on souhaite recevoir une réponse sous la forme de texte
            out.println("2");

            // Déclaration de la variable permettant d'écrire
            // dans le buffer de sortie de la socket
            BufferedReader bis = new BufferedReader(new InputStreamReader(mySocket.getInputStream()));

            // Lecture de la réponse renvoyée par le serveur
            inputLine = bis.readLine();
        }

        // Si la connection échoue
        else
        {
            // Message d'information
            Toast.makeText(MainActivity.super.getApplicationContext(), "La connexion au serveur a échouée", Toast.LENGTH_SHORT).show();
        }

        // Fermeture de la connection socket
        mySocket.close();

        // On retourne la valeur lue
        return inputLine;
    }

    // Fonction permettant d'envoyer une photo
    // au serveur EN COURS D'IMPLEMENTATION
    private void sendImageToServer() throws IOException {
        // Permission pour la création d'un connection via un socket
        StrictMode.ThreadPolicy policy = new StrictMode.ThreadPolicy.Builder().permitAll().build();
        StrictMode.setThreadPolicy(policy);

        // Récupération de l'adresse IP du serveur
        String myHost = serverIP1.toString() + "." + serverIP2.toString() + "." + serverIP3.toString() + "." + serverIP4.toString();

        // Déclaration et initialisation de la chaine
        // de caractère permettant de récupérer le message du serveur
        String inputLine = "";

        // Création de la socket et connection
        Socket mySocket = new Socket(myHost, serverPort);

        // Si la connection est réussi
        if (mySocket.isConnected() == true) {
            // Message d'information
            Toast.makeText(MainActivity.super.getApplicationContext(), "Connexion au serveur réussi", Toast.LENGTH_SHORT).show();

            // Déclaration du buffer de sortie du socket
            OutputStream outstream = mySocket.getOutputStream();

            // Déclaration de la variable permettant d'écrire
            // dans le buffer de sortie de la socket
            PrintWriter out = new PrintWriter(outstream, true);

            // Écriture de la valeur "3" dans le buffer de sortie afin d'indiquer
            // que l'on souhaite envoyer une photographie
            out.println("3");

            // Déclaration de la variable permettant d'écrire
            // dans le buffer de sortie de la socket
            BufferedReader bis = new BufferedReader(new InputStreamReader(mySocket.getInputStream()));

            // Lecture de la réponse renvoyée par le serveur
            inputLine = bis.readLine();
        }

        // Si la connection échoue
        else {
            // Message d'information
            Toast.makeText(MainActivity.super.getApplicationContext(), "La connexion au serveur a échouée", Toast.LENGTH_SHORT).show();
        }

        // Fermeture de la connection socket
        mySocket.close();

    }

    // Fonction permettant de générer un fichier
    // JPEG lors de la prise de photo et de
    // l'enregistrer
    Camera.PictureCallback mPicture = new Camera.PictureCallback() {
        @Override
        public void onPictureTaken(byte[] data, Camera camera) {
            File pictureFile = getOutputMediaFile();

            // Affectation de imageFile avec le fichier
            // image de la photo prise
            imageFile = pictureFile;
            if (pictureFile == null) {
                return;
            }
            try {
                // Enregistrement de la photo
                FileOutputStream fos = new FileOutputStream(pictureFile);
                fos.write(data);
                fos.close();
            } catch (FileNotFoundException e) {

            } catch (IOException e) {
            }
        }
    };

    // Fonction permettant de creer un fichier
    // pour stocker une photo prise
    private static File getOutputMediaFile() {
        File mediaStorageDir = new File(
                Environment
                        .getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                "FaceFinderApp");
        if (!mediaStorageDir.exists()) {
            if (!mediaStorageDir.mkdirs()) {
                Log.d("MyCameraApp", "failed to create directory");
                return null;
            }
        }
        // Creation d'un nom de fichier unique
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss")
                .format(new Date());
        File mediaFile;
        mediaFile = new File(mediaStorageDir.getPath() + File.separator
                + "IMG_" + timeStamp + ".jpg");
        return mediaFile;
    }


    /**
     * ATTENTION: This was auto-generated to implement the App Indexing API.
     * See https://g.co/AppIndexing/AndroidStudio for more information.
     */
    public Action getIndexApiAction() {
        Thing object = new Thing.Builder()
                .setName("Main Page") // TODO: Define a title for the content shown.
                // TODO: Make sure this auto-generated URL is correct.
                .setUrl(Uri.parse("http://[ENTER-YOUR-URL-HERE]"))
                .build();
        return new Action.Builder(Action.TYPE_VIEW)
                .setObject(object)
                .setActionStatus(Action.STATUS_TYPE_COMPLETED)
                .build();
    }

    @Override

    // Fonction exécutée au démarrage de l'application
    public void onStart() {
        super.onStart();
        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        client.connect();
        AppIndex.AppIndexApi.start(client, getIndexApiAction());
    }

    @Override

    // Fonction exécutée à l'arrêt de l'application
    public void onStop() {
        super.onStop();

        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        AppIndex.AppIndexApi.end(client, getIndexApiAction());
        client.disconnect();
    }
}