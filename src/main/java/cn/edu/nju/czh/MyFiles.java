package cn.edu.nju.czh;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;

public class MyFiles {
    ArrayList<String> files = new ArrayList<>();
    String inputDir = null;
    String outputDir = null;
    int index = 0;

    public void setInputDir(String inputDir) {
        this.inputDir = inputDir;
    }

    public void setOutputDir(String outputDir) {
        this.outputDir = outputDir;
    }

    public void init() throws FileNotFoundException{
        if(inputDir == null) {
            throw new FileNotFoundException();
        }
        else {
            File file = new File(inputDir);
            String[] fileNames = file.list();
            for(String file1 : fileNames) {
                files.add(file1.split("[.]")[0]);
            }
        }

        if(outputDir == null) {
            throw new FileNotFoundException();
        }
        else {
            File file = new File(outputDir);
            for(File file1 : file.listFiles()) {
                file1.delete();
            }
        }
    }

    public ArrayList<String> getFiles() {
        return files;
    }

}
