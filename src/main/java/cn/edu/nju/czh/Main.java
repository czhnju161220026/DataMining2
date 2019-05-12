package cn.edu.nju.czh;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;

/**
 * Hello world!
 */

public class Main {
    public static List<MyClassifier> generateClassifiers() {
        ArrayList<MyClassifier> classifiers = new ArrayList<>();
        //添加普通分类器
        classifiers.add(new MyClassifier(new J48(),"J48"));
        classifiers.add(new MyClassifier(new NaiveBayes(),"NaiveBayes"));
        classifiers.add(new MyClassifier(new SMO(), "SVM"));
        classifiers.add(new MyClassifier(new MultilayerPerceptron(), "Neural Network"));
        classifiers.add(new MyClassifier(new IBk(),"KNN"));

        //添加集成后的分类器
        Bagging bagging = new Bagging();
        bagging.setClassifier(new J48());
        classifiers.add(new MyClassifier(bagging, "Bagging J48"));

        bagging = new Bagging();
        bagging.setClassifier(new NaiveBayes());
        classifiers.add(new MyClassifier(bagging, "Bagging NaiveBayes"));

        bagging = new Bagging();
        bagging.setClassifier(new SMO());
        classifiers.add(new MyClassifier(bagging, "Bagging SVM"));

        bagging = new Bagging();
        bagging.setClassifier(new MultilayerPerceptron());
        classifiers.add(new MyClassifier(bagging, "Bagging Neural Network"));

        bagging = new Bagging();
        bagging.setClassifier(new IBk());
        classifiers.add(new MyClassifier(bagging, "Bagging KNN"));

        return classifiers;
    }

    public static void main(String[] args) {
        String inputDir = "data";
        String outputDir = "result";
        String outputFileSuffix = "_result.txt";
        MyFiles inputFiles = new MyFiles();
        inputFiles.setInputDir(inputDir);
        inputFiles.setOutputDir(outputDir);

        try {
            inputFiles.init();
            List<String> files = inputFiles.getFiles();
            System.out.println("File loaded.");
            List<MyClassifier> classifiers = generateClassifiers();
            System.out.println("Classifiers generated.");
            for(MyClassifier classifier : classifiers) {
                System.out.print("# Validating "+classifier.getName() + ". ");
                Date begin = new Date();
                for(String file : files) {
                    MyEvaluation.tenCrossValidate(classifier,inputDir + "/"+file+".arff", outputDir + "/"+file+outputFileSuffix);
                }
                Date end = new Date();
                long second = (end.getTime() - begin.getTime()) / 1000; //秒
                long minute = second / 60;
                second = second % 60;
                System.out.println("Validation completed. Time cost:"+minute+"m"+second+"s.");
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }
}
