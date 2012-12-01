package net.sukharevd.hadoop.normalization;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class NormalizerDriver extends Configured implements Tool {

    private List<Map<String, Long>> columnLabelsReplacements = new ArrayList<Map<String, Long>>();

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new NormalizerDriver(), args);
        System.exit(res);
    }
    
    @Override
    public int run(String[] arg0) throws Exception {
        Configuration conf = getConf();
        JobConf job = new JobConf(conf, NormalizerDriver.class);
        normalizeDataSet(conf, job, new Path("/home/dmitriy/Desktop/data-sets/kddcup.data_10_percent_corrected"));
        return 0;
    }

    private void normalizeDataSet(Configuration conf, JobConf job, Path in) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        FSDataInputStream inputStream = fs.open(in);
        BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
        Path numerizedInputPath = in.suffix(".numerized");
        FSDataOutputStream out = fs.create(numerizedInputPath);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(out));

        System.out.println("Eliminate nominal values");
        while (true) {
            String line = br.readLine();
            if (line == null)
                break;
            line = line.substring(0, line.length() - 1); // remove the point
            String[] split = line.split(",");
            eliminateNominalValues(split);
            StringBuilder builder = new StringBuilder(split[0]);
            for (int i = 1; i < split.length; i++) {
                builder.append('\t').append(split[i]);
            }
            builder.append('\n');
            bw.write(builder.toString());
        }
        br.close();
        bw.flush();
        bw.close();

        Path InputFeaturesMapPath = in.suffix(".featuresMap");
        out = fs.create(InputFeaturesMapPath);
        out.writeUTF(columnLabelsReplacements.toString());
        out.close();

        System.out.println("Means");
        long counter = 0;
        Jama.Matrix means = null;
        inputStream = fs.open(numerizedInputPath);
        br = new BufferedReader(new InputStreamReader(inputStream));
        while (true) {
            String line = br.readLine();
            if (line == null)
                break;
            String[] split = line.split("\t");
            if (means == null) {
                means = new Jama.Matrix(1, split.length);
            }
            Jama.Matrix vector = convertToVector(split);
            System.out.println(counter + ": " + means.getColumnDimension() + "x" + means.getRowDimension());
            System.out.println(counter + ": " + vector.getColumnDimension() + "x" + vector.getRowDimension());
            means = means.plus(vector);
            counter++;
        }
        means.timesEquals(1d / counter);
        br.close();
        means.set(0, means.getColumnDimension()-1, 0); // don't wanna change labels

        System.out.println("Sigmas");
        Jama.Matrix sigmas = new Jama.Matrix(1, means.getColumnDimension());
        inputStream = fs.open(numerizedInputPath);
        br = new BufferedReader(new InputStreamReader(inputStream));
        while (true) {
            String line = br.readLine();
            if (line == null)
                break;
            String[] split = line.split("\t");
            Jama.Matrix vector = convertToVector(split);
            sigmas.plusEquals(vector.minus(means).arrayTimes(vector.minus(means)));
        }
        sigmas.timesEquals(1d / counter);
        sigmas.set(0, means.getColumnDimension()-1, 1d); // don't wanna change labels
        for (int i = 0; i < sigmas.getArray()[0].length; i++) {
            if (sigmas.get(0, i) == 0) {
                sigmas.set(0, i, 1d);
                System.out.println("WARN: sigma of feature " + i + " is zero, mean is " + means.get(0, i));
            }
        }

        for (int i = 0; i < sigmas.getColumnDimension(); i++) {
            sigmas.set(0, i, Math.sqrt(sigmas.get(0, i)));
        }
        br.close();
        for (int i = 0; i < means.getArray().length; i++) {
            System.out.println("Means: " + Arrays.toString(means.getArray()[i]));
        }
        for (int i = 0; i < means.getArray().length; i++) {
            System.out.println("Sigmas: " + Arrays.toString(sigmas.getArray()[i]));
        }

        System.out.println("Normalized");
        inputStream = fs.open(numerizedInputPath);
        br = new BufferedReader(new InputStreamReader(inputStream));
        Path normalizedInputPath = in.suffix(".normalized");
        out = fs.create(normalizedInputPath);
        bw = new BufferedWriter(new OutputStreamWriter(out));
        while (true) {
            String line = br.readLine();
            if (line == null)
                break;
            String[] split = line.split("\t");
            Jama.Matrix vector = convertToVector(split);
            vector.minusEquals(means).arrayRightDivideEquals(sigmas);
            StringBuilder builder = new StringBuilder(Double.toString(vector.get(0, 0)));
            for (int i = 1; i < split.length; i++) {
                builder.append('\t').append(vector.get(0, i));
            }
            bw.write(builder.toString());
            bw.newLine();
        }
        br.close();
        bw.flush();
        bw.close();
    }

    private Jama.Matrix convertToVector(String[] split) {
        double[] vectorArray = new double[split.length];
        for (int i = 0; i < split.length; i++) {
            vectorArray[i] = Double.valueOf(split[i]);
        }
        return new Jama.Matrix(new double[][] { vectorArray });
    }

    private void eliminateNominalValues(String[] split) {
        if (columnLabelsReplacements.isEmpty()) {
            for (int i = 0; i < split.length; i++) {
                columnLabelsReplacements.add(new HashMap<String, Long>());
            }
        }
        for (int i = 0; i < split.length; i++) {
            if (!split[i].matches("[0-9\\.]+")) {
                if (columnLabelsReplacements.get(i).get(split[i]) != null) {
                    split[i] = Long.valueOf(columnLabelsReplacements.get(i).get(split[i])).toString();
                } else {
                    long last = 0;
                    for (Long value : columnLabelsReplacements.get(i).values()) {
                        if (last <= value) {
                            last = value + 1;
                        }
                    }
                    columnLabelsReplacements.get(i).put(split[i], last);
                    split[i] = Long.valueOf(last).toString();
                }
            }
        }
    }

}
