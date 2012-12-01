package net.sukharevd.hadoop.nn.services;

import java.io.IOException;
import java.util.Arrays;

import javax.ws.rs.Path;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.UriInfo;

import net.sukharevd.hadoop.entities.Matrix;
import net.sukharevd.hadoop.nn.NeuralNetworkPredictor;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

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
        org.apache.hadoop.fs.Path file = new org.apache.hadoop.fs.Path("/user/dmitriy/nn/outputs/output13/it1/part-00001");
        SequenceFile.Reader thetasReader = new SequenceFile.Reader(fs, file, conf);
        IntWritable layerId = new IntWritable();
        Text text = new Text();
        StringBuilder builder = new StringBuilder();
        while (thetasReader.next(layerId, text)) {
            builder.append(text.toString());
        }
        Jama.Matrix thetas1 = new Jama.Matrix(Matrix.valueOf(builder.toString().split("\t")[0]).getItems());
        thetasReader.close();
        
        file = new org.apache.hadoop.fs.Path("/user/dmitriy/nn/outputs/output13/it1/part-00002");
        thetasReader = new SequenceFile.Reader(fs, file, conf);
        layerId = new IntWritable();
        text = new Text();
        builder = new StringBuilder();
        while (thetasReader.next(layerId, text)) {
            builder.append(text.toString());
        }
        Jama.Matrix thetas2 = new Jama.Matrix(Matrix.valueOf(builder.toString().split("\t")[0]).getItems());
        // use elimination of nominal values here
        Jama.Matrix example = new Jama.Matrix(Matrix.valueOf(value.replace(',', '\t').replace(".", "")).getItems());
        // use normalization here.
        int predictedY = NeuralNetworkPredictor.predict(Arrays.asList(thetas1, thetas2), example);
        return Response.ok(predictedY).build();
    }    
    
}
