package net.sukharevd.hadoop.nn;

import java.io.IOException;
import java.util.Iterator;

import net.sukharevd.hadoop.entities.Matrix;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

public class NeuralNetworkCombiner extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text> {
    
    private Matrix gradWritable = new Matrix();

    @Override
    public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter)
            throws IOException {
        Jama.Matrix result = null;
        long counter = 0;
        double J = 0d;
        while (values.hasNext()) {
            String value = values.next().toString();
            String[] split = value.split("\t");
            String matrixi = split[0];
            String Ji = split[1];
            String amounti = split[2];
            Matrix matrix = Matrix.valueOf(matrixi);
            J += Double.parseDouble(Ji);
            if (result == null) {
                result = new Jama.Matrix(matrix.getItems());
            } else {
                result.plusEquals(new Jama.Matrix(matrix.getItems()));
            }
            counter += Long.parseLong(amounti);
        }
        gradWritable.setItems(result.getArray());
        output.collect(key, new Text(gradWritable.toString() + "\t" + J + "\t" + counter));
    }
}