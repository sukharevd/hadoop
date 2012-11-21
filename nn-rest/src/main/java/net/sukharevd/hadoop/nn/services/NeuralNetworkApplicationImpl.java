package net.sukharevd.hadoop.nn.services;

import java.util.HashSet;
import java.util.Set;

import javax.ws.rs.core.Application;

public class NeuralNetworkApplicationImpl extends Application {
    @Override
    public Set<Class<?>> getClasses() {
        Set<Class<?>> set = new HashSet<Class<?>>();
        set.add(NeuralNetworkServiceImpl.class);
        return set;
    }
}
