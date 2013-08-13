package net.sukharevd.hadoop.nn.services;

import java.io.IOException;

import javax.ws.rs.FormParam;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

/** The interface for neural network classifier REST service. */
public interface NeuralNetworkService {

    @POST
    @Produces({ MediaType.TEXT_PLAIN })
    /** Returns predicted class for the specified record. */
    Response predict(String value) throws IOException;
    
    @POST
    @Produces( MediaType.TEXT_PLAIN )
    @Path("statistics")
    Response statistics(@FormParam("directory") String directory) throws IOException;
}
