package cn.edu.nju.czh;

import jdk.nashorn.internal.runtime.regexp.joni.ScanEnvironment;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.io.*;
import java.util.Random;
import java.util.Scanner;

public class MyEvaluation {
    public static void tenCrossValidate(MyClassifier myClassifier, String fileName, String resultFile) {
        try {
            // 获取训练街
            Instances instances = new Instances(new BufferedReader(new FileReader(fileName)));
            instances.setClassIndex(instances.numAttributes() - 1);
            Evaluation evaluation = new Evaluation(instances);
            // 十折交叉验证
            evaluation.crossValidateModel(myClassifier.getClassifier(),instances,10,new Random(1));
            String summary = evaluation.toSummaryString("==== " + myClassifier.getName()+" summary ====",false);
            String detail = evaluation.toClassDetailsString("==== "+ myClassifier.getName() + " detail ====");
            // 将结果写入指定文件
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(resultFile),true));
            bufferedWriter.write(summary);
            bufferedWriter.write(detail);
            bufferedWriter.newLine();
            bufferedWriter.write("=================================================================");
            bufferedWriter.newLine();
            bufferedWriter.flush();
            bufferedWriter.close();

        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(-1);
        }
        catch (IOException e) {
            e.printStackTrace();
            System.exit(-2);
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(-3);
        }
    }
}
