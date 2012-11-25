package net.sukharevd.hadoop.nn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;

import net.sukharevd.hadoop.entities.Matrix;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;

public class NeuralNetworkUtils {
    
    public static List<Jama.Matrix> readThetasFromThetasDir(JobConf job) {
        assert (job.get("thetas.path") != null);
        String filenames = "";
        try {
            FileSystem fs = FileSystem.get(job);
            Path centroidsPath = new Path(job.get("thetas.path"));
            if (fs.isFile(centroidsPath)) {
                filenames += centroidsPath.toString();
                return new ArrayList<Jama.Matrix>(readFile(job, fs, centroidsPath).values());
            } else { // isDirectory
                FileStatus[] listStatus = fs.listStatus(centroidsPath);
                List<Jama.Matrix> matrices = new ArrayList<Jama.Matrix>();
                for (FileStatus fileStatus : listStatus) {
                    if (fileStatus.getPath().toString().contains("part-")) {
                        filenames += fileStatus.getPath().toString();
                        SortedMap<Integer, Jama.Matrix> matrixMap = new TreeMap<Integer, Jama.Matrix>();
                        matrixMap.putAll(readFile(job, fs, fileStatus.getPath()));
                        matrices.addAll(matrixMap.values());
                    }
                }
                return matrices;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        throw new IllegalStateException("Read files: [" + filenames + "] from [" + job.get("thetas.path") + "]. See previous exceptions");
    }

    public static SortedMap<Integer, Jama.Matrix> readFile(JobConf job, FileSystem fs, Path thetasPath) throws IOException {
        SequenceFile.Reader thetasReader = new SequenceFile.Reader(fs, thetasPath, job);
        IntWritable layerId = new IntWritable();
        Text text = new Text();
        Matrix matrix = new Matrix();
        SortedMap<Integer, Jama.Matrix> matrices = new TreeMap<Integer, Jama.Matrix>();
        while (thetasReader.next(layerId, text)) {
            matrix = Matrix.valueOf(text.toString().split("\t")[0]);
            matrices.put(layerId.get(), new Jama.Matrix(matrix.getItems()));
        }
        assert (matrices.size() > 1);
        assert (!matrices.get(0).equals(matrices.get(1))); // assumption that there's at least one hidden layer
        thetasReader.close();
        return matrices;
    }
    
    public static Jama.Matrix sigmoidGradient(Jama.Matrix out) {
        double[][] oneArray = new double[out.getRowDimension()][out.getColumnDimension()];
        for (double[] oneVector : oneArray) {
            Arrays.fill(oneVector, 1d);
        }
        Jama.Matrix one = new Jama.Matrix(oneArray);
        return sigmoid(out).arrayTimes(one.minus(sigmoid(out)));
    }

    public static Jama.Matrix sigmoid(Jama.Matrix matrix) {
        Jama.Matrix result = matrix.copy();
        for (int i = 0; i < result.getRowDimension(); i++) {
            for (int j = 0; j < result.getColumnDimension(); j++) {
                result.set(i, j, sigmoid(result.get(i, j)));
            }
        }
        return result;
    }

    public static double sigmoid(double z) {
        return 1d / (1d + Math.exp(-z));
    }

}
