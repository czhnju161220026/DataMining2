package cn.edu.nju.czh;

import weka.classifiers.Classifier;

public class MyClassifier {
    private Classifier classifier;
    private String name;

    public MyClassifier(Classifier classifier, String name) {
        this.classifier = classifier;
        this.name = name;
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public void setClassifier(Classifier classifier) {
        this.classifier = classifier;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
