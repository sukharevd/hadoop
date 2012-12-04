package net.sukharevd.hadoop.nn.services;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.ws.rs.Path;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.UriInfo;

import net.sukharevd.hadoop.entities.Matrix;
import net.sukharevd.hadoop.nn.NeuralNetworkPredictor;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

/** The class that represents neural network classifier REST service. */
@Path("nn")
public class NeuralNetworkServiceImpl implements NeuralNetworkService {

    @Context UriInfo uriInfo;

    @Override
    public Response predict(String value) throws IOException {
        Configuration conf = new Configuration();
        conf.addResource(new org.apache.hadoop.fs.Path("/opt/hadoop/conf/core-site.xml"));
        conf.addResource(new org.apache.hadoop.fs.Path("/opt/hadoop/conf/hdfs-site.xml"));

        FileSystem fs = FileSystem.get(conf);
        Jama.Matrix thetas1 = getMatrixFromSequenceFile(conf, fs, "/user/dmitriy/nn/outputs/output68/it2/part-00000");
        Jama.Matrix thetas2 = getMatrixFromSequenceFile(conf, fs, "/user/dmitriy/nn/outputs/output68/it2/part-00001");
        String[] exampleSplit = value.replace(',', '\t').split("\t");
        Jama.Matrix means = getMatrixFromTextFile(fs, "/user/dmitriy/nn/inputs/input_10p.csv.means");
        Jama.Matrix sigmas = getMatrixFromTextFile(fs, "/user/dmitriy/nn/inputs/input_10p.csv.sigmas");
        List<Map<String, Long>> nominal2numericMaps = getNominalMaps(conf, fs, "/user/dmitriy/nn/inputs/input_10p.csv.featuresMap");
        //if (exampleSplit.length + 1 != nominal2numericMaps.size()) throw new IllegalArgumentException();
        //if (exampleSplit.length + 1 != means.getColumnDimension()) throw new IllegalArgumentException();
        //if (exampleSplit.length + 1 != sigmas.getColumnDimension()) throw new IllegalArgumentException();
        double[][] exampleArray = new double[1][exampleSplit.length];
        for (int i = 0; i < exampleSplit.length; i++) {
            if (nominal2numericMaps.get(i) != null && nominal2numericMaps.get(i).get(exampleSplit[i]) != null) {
                exampleSplit[i] = nominal2numericMaps.get(i).get(exampleSplit[i]).toString();
            }
            exampleArray[0][i] = Double.parseDouble(exampleSplit[i]);
        }
        Jama.Matrix example = new Jama.Matrix(exampleArray);
        example.minusEquals(means).arrayRightDivideEquals(sigmas);
        int predictedY = NeuralNetworkPredictor.predict(Arrays.asList(thetas1, thetas2), example);
        Map<String, Long> labelsMap = nominal2numericMaps.get(nominal2numericMaps.size()-1);
        for (String key : labelsMap.keySet()) {
            if (labelsMap.get(key).equals((long)predictedY)) {
                return Response.ok(key).build();
            }
        }
        return Response.status(500).entity("Unknown predicted class: " + predictedY).build();
    }

    private List<Map<String, Long>> getNominalMaps(Configuration conf, FileSystem fs, String path) throws IOException {
        org.apache.hadoop.fs.Path file = new org.apache.hadoop.fs.Path(path);
        SequenceFile.Reader thetasReader = new SequenceFile.Reader(fs, file, conf);
        NullWritable nullWritable = NullWritable.get();
        MapWritable mapWritable = new MapWritable();
        List<Map<String, Long>> resultMaps = new ArrayList<Map<String, Long>>();
        while (thetasReader.next(nullWritable, mapWritable)) {
            HashMap<String, Long> map = new HashMap<String, Long>();
            for (Writable key : mapWritable.keySet()) {
                map.put(((Text) key).toString(), ((LongWritable) mapWritable.get(key)).get());
            }
            resultMaps.add(map);
        }
        return resultMaps;
    }

    private Jama.Matrix getMatrixFromTextFile(FileSystem fs, String path) throws IOException {
        org.apache.hadoop.fs.Path file = new org.apache.hadoop.fs.Path(path);
        FSDataInputStream is = fs.open(file);
        String matrixString = is.readUTF();
        is.close();
        Jama.Matrix thetas1 = new Jama.Matrix(Matrix.valueOf(matrixString.split("\t")[0]).getItems());
        return thetas1;
    }
    
    private Jama.Matrix getMatrixFromSequenceFile(Configuration conf, FileSystem fs, String path) throws IOException {
        org.apache.hadoop.fs.Path file = new org.apache.hadoop.fs.Path(path);
        SequenceFile.Reader thetasReader = new SequenceFile.Reader(fs, file, conf);
        IntWritable layerId = new IntWritable();
        Text text = new Text();
        StringBuilder builder = new StringBuilder();
        while (thetasReader.next(layerId, text)) {
            builder.append(text.toString());
        }
        Jama.Matrix thetas1 = new Jama.Matrix(Matrix.valueOf(builder.toString().split("\t")[0]).getItems());
        thetasReader.close();
        return thetas1;
    }    
    
}
