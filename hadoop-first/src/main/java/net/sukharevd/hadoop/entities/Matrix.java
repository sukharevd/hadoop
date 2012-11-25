package net.sukharevd.hadoop.entities;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.io.Writable;

public class Matrix implements Writable {
    
    private double[][] items;

    public Matrix() {
    }
    
    public Matrix(double[][] items) {
        this.items = Arrays.copyOf(items, items.length); // not deep clone
    }

    public Matrix(Matrix clone) {
        items = Arrays.copyOf(clone.items, clone.items.length); // not deep clone
    }

    public double[][] getItems() {
        return items;
    }

    public void setItems(double[][] items) {
        this.items = items;
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        String matrixStr = in.readUTF();
        Matrix matrix = valueOf(matrixStr);
        items = matrix.getItems();
    }

    public static Matrix valueOf(String matrixStr) {
        List<List<Double>> dynamicItems = new ArrayList<List<Double>>();
        String[] split1 = matrixStr.split("\\s*;\\s*");
        for (String vectorString : split1) {
            String[] split2 = vectorString.split("\\s+");
            ArrayList<Double> vector = new ArrayList<Double>(split2.length+1);
            for (String vectorItemString : split2) {
                vector.add(Double.valueOf(vectorItemString));
            }
            dynamicItems.add(vector);
        }
        double[][] items = new double[dynamicItems.size()][dynamicItems.get(0).size()];
        for (int i = 0; i < dynamicItems.size(); i++) {
            for (int j = 0; j < dynamicItems.get(i).size(); j++) {
                items[i][j] = dynamicItems.get(i).get(j);
            }
        }
        return new Matrix(items);
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeUTF(this.toString());
    }

    private String vector2String(double[] vector) {
        StringBuilder vectorStr = new StringBuilder();
        if (vector.length > 0) {
            vectorStr.append(vector[0]);
        }
        for (int i = 1; i < vector.length; i++) {
            vectorStr.append(' ').append(vector[i]);
        }
        return vectorStr.toString();
    }

    @Override
    public String toString() {
        StringBuilder itemsStr = new StringBuilder();
        if (items.length > 0) {
            itemsStr.append(vector2String(items[0]));
        }
        for (int i = 1; i < items.length; i++) {
            itemsStr.append(';').append(vector2String(items[i]));
        }
        return itemsStr.toString();
    }

}