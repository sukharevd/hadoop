package net.sukharevd.hadoop.nn;

import static net.sukharevd.hadoop.nn.NeuralNetworkUtils.*;
import static net.sukharevd.hadoop.util.JamaExt.*;

import java.util.List;

public class NeuralNetworkPredictor {
    public static int predict(List<Jama.Matrix> thetas, Jama.Matrix example) {
        Jama.Matrix x = example.getMatrix(0, example.getRowDimension()-1, 0, example.getColumnDimension()-2);
        Jama.Matrix[] a = new Jama.Matrix[thetas.size() + 1];
        Jama.Matrix[] z = new Jama.Matrix[thetas.size() + 1];
        z[0] = x;
        a[0] = x;
        for (int i = 1; i <= thetas.size(); i++) {
            a[i-1] = addOneColumn(a[i-1]);
            z[i] = a[i-1].times(thetas.get(i-1).transpose());
            a[i] = sigmoid(z[i]);
        }
        Jama.Matrix out = a[a.length - 1];
        return indexMax(out);
    }
}
