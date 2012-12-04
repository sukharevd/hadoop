package net.sukharevd.hadoop.nn;

import java.io.IOException;

import org.glassfish.grizzly.http.server.HttpServer;

import com.sun.jersey.api.container.grizzly2.GrizzlyServerFactory;
import com.sun.jersey.api.core.PackagesResourceConfig;
import com.sun.jersey.api.core.ResourceConfig;

/** Точка входу, що запускає Grizzly веб-серер та розташовує на ньому веб-сервіс книжок. */
public class Main {

    public static final String BASE_URI = "http://localhost:9998";

    public static void main(String[] args) throws IOException {
        ResourceConfig rc = new PackagesResourceConfig("net.sukharevd.hadoop.nn");
        HttpServer httpServer = GrizzlyServerFactory.createHttpServer(BASE_URI, rc);
        System.out.println(String.format("Jersey app started at %s. Hit enter to stop it...", BASE_URI));
        System.in.read();
        httpServer.stop();
    }
}
