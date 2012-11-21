package net.sukharevd.hadoop.nn.services;

import java.io.IOException;

import javax.ws.rs.Path;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.UriInfo;

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
        return Response.ok(builder.toString()).build();
    }    
    
}
