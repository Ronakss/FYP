package com.example.app.db;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import com.example.app.model.ManagersModel;
import com.example.app.model.ProjectsModel;

import java.util.ArrayList;

public class AppDataBase {
    /*DATA BASE NAME AND VERSION , CHANGE VERSION WHILE CHANGE IN TABLE*/
    // Reference: The following code is from
    // https://guides.codepath.com/android/local-databases-with-sqliteopenhelper#inserting-new-records
    //Our database name is managment
    //Here we have AppDataBase class, in this class about database SQLite to create a table for manager
   // info and projects info. such and id, name, etc. in this class we have tables, insert, queries for
     //both tables project and manager. Then in this class, we also have deleted,
    //for example, if user deletes data in the app. Then we also have SQLite database drop table

    private static final String DATA_BASE_NAME = "managment.db";
    private static final int DATA_BASE_VERSION = 1;
    /*Here we have table names for managers and projects   */
    private static final String TABLE_NAME_MANAGERS = "managers";
    private static final String TABLE_NAME_PROJECTS = "projects";

    /* this line of code about managers  column which id and name columns of manager table table*/
    // reference : https://stackoverflow.com/questions/48161646/cannot-resolve-symbol-symbol-name-private-field-column-name-is-never-used
    private static final String COLUMN_ID = "id";
    private static final String COLUMN_TITLE = "name";


    /* This line of code is columns of  projects table which includes id  manager id ,projects name and comments
    * this will shown in screen when we going add and asaign name ect */
    private static final String COLUMN_ITEM_ID = "id";
    private static final String COLUMN_MGR_ID = "manager_id";
    private static final String COLUMN_PROJECT = "project_name";
    private static final String COLUMN_COMENTS = "comments";


    /* this line of code about drop tables for both managers and proejcts  queries*/

    private static final String DROP_TABLE_MANAGERS = "DROP TABLE IF EXISTS " + TABLE_NAME_MANAGERS;
    private static final String DROP_TABLE_PROJECTS = "DROP TABLE IF EXISTS " + TABLE_NAME_PROJECTS;



    /* this line of code we are create  table queries for manager and primary key */


    private static final String CREATE_TABLE_MANAGERS =
            "CREATE TABLE " + TABLE_NAME_MANAGERS + " (" +
                    COLUMN_ID + " INTEGER PRIMARY KEY," +
                    COLUMN_TITLE + " TEXT)";

// here I create table for project which includes item ID MANAGER ID project and comments
    private static final String CREATE_TABLE_PROJECTS =
            "CREATE TABLE " + TABLE_NAME_PROJECTS + " (" +
                    COLUMN_ITEM_ID + " INTEGER PRIMARY KEY," +
                    COLUMN_MGR_ID + " TEXT," +
                    COLUMN_PROJECT + " TEXT," +
                    COLUMN_COMENTS + " TEXT)";


    private Context context;
    private SQLiteDatabase sqLiteDatabase;

    private Db Db;

    public AppDataBase(Context context) {
        this.context = context;
    }


    public long insertManager(String name) {
        ContentValues contentValues = new ContentValues();
        contentValues.put(COLUMN_TITLE, name);


        long insertedId = 0;
        try {
            insertedId = sqLiteDatabase.insert(TABLE_NAME_MANAGERS, null, contentValues);

        } catch (Exception e) {

        }
        if (insertedId < 0) {
            insertedId = 0;
        }
        return insertedId;
    }

    public long insertProject(String manager_id, String project_name, String comments) {
        ContentValues contentValues = new ContentValues();
        contentValues.put(COLUMN_MGR_ID, manager_id);
        contentValues.put(COLUMN_PROJECT, project_name);
        contentValues.put(COLUMN_COMENTS, comments);

        long insertedId = 0;
        try {
            insertedId = sqLiteDatabase.insert(TABLE_NAME_PROJECTS, null, contentValues);

        } catch (Exception e) {

        }
        if (insertedId < 0) {
            insertedId = 0;
        }
        return insertedId;
    }


    public ArrayList<ManagersModel> getAllManagers() {

        ArrayList<ManagersModel> myList = new ArrayList<>();
        Cursor cursor = sqLiteDatabase.query(TABLE_NAME_MANAGERS, null, null, null, null, null, null);
        while (cursor.moveToNext()) {


            String id = cursor.getString(cursor.getColumnIndex(COLUMN_ID));
            String name = cursor.getString(cursor.getColumnIndex(COLUMN_TITLE));
            ManagersModel UsersModel = new ManagersModel(id, name);
            myList.add(UsersModel);
        }

        return myList;
    }

    public ArrayList<ProjectsModel> getAllProject(String manager_id) {

        ArrayList<ProjectsModel> myList = new ArrayList<>();
        Cursor cursor = sqLiteDatabase.query(TABLE_NAME_PROJECTS, null, null, null, null, null, null);
        while (cursor.moveToNext()) {
            String managerid = cursor.getString(cursor.getColumnIndex(COLUMN_MGR_ID));


            /*this check is to sort projects according to managers id*/
            if (manager_id.equals(managerid)) {
                String id = cursor.getString(cursor.getColumnIndex(COLUMN_ID));
                String name = cursor.getString(cursor.getColumnIndex(COLUMN_PROJECT));
                String coments = cursor.getString(cursor.getColumnIndex(COLUMN_COMENTS));

                ProjectsModel UsersModel = new ProjectsModel(id, managerid, name, coments);
                myList.add(UsersModel);
            }


        }

        return myList;
    }

    public long updateItemById(String id, String title) {
        ContentValues contentValues = new ContentValues();
        contentValues.put(COLUMN_TITLE, title);
        long insertedId = sqLiteDatabase.update(TABLE_NAME_MANAGERS, contentValues, COLUMN_ID + " = ?", new String[]{id});


        return insertedId;
    }


    public long updateProjectByIdInList(String id, String project, String comments) {
        ContentValues contentValues = new ContentValues();
        contentValues.put(COLUMN_PROJECT, project);
        contentValues.put(COLUMN_COMENTS, comments);
        long insertedId = sqLiteDatabase.update(TABLE_NAME_PROJECTS, contentValues, COLUMN_ID + " = ?", new String[]{id});


        return insertedId;
    }

    public boolean deleteManager(String id) {
        return sqLiteDatabase.delete(TABLE_NAME_MANAGERS, COLUMN_ID + "=?", new String[]{id}) > 0;
    }


    public boolean deleteProject(String id) {
        return sqLiteDatabase.delete(TABLE_NAME_PROJECTS, COLUMN_ID + "=?", new String[]{id}) > 0;
    }


    public AppDataBase open() throws android.database.SQLException {
        try {
            Db = new Db(context);
            sqLiteDatabase = Db.getWritableDatabase();

        } catch (Exception e) {

        }
        return AppDataBase.this;
    }

    public void close() {
        Db.close();
    }

    private class Db extends SQLiteOpenHelper {

        public Db(Context context) {
            super(context, DATA_BASE_NAME, null, DATA_BASE_VERSION);
        }

        @Override
        public void onCreate(SQLiteDatabase sqLiteDatabase) {
            /*these methods call only on  first install or on db version upgrade*/
            
            sqLiteDatabase.execSQL(CREATE_TABLE_MANAGERS);
            sqLiteDatabase.execSQL(CREATE_TABLE_PROJECTS);

        }

        @Override
        // reference : https://stackoverflow.com/questions/41437355/sqlitedatabase-object
        public void onUpgrade(SQLiteDatabase sqLiteDatabase, int i, int i1) {
            /*these methods call only on  first install or on db version upgrade*/

            sqLiteDatabase.execSQL(DROP_TABLE_MANAGERS);
            sqLiteDatabase.execSQL(DROP_TABLE_PROJECTS);

        }
    }

}
