package net.sukharevd.hadoop.util;

import java.util.Arrays;

public final class JamaExt {
    
    private JamaExt() {
        // to avoid creation of utility class.
    }
    
    public static double sum(Jama.Matrix matrix) {
        double result = 0d;
        double[][] array = matrix.getArray();
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[0].length; j++) {
                result += array[i][j];
            }
        }
        return result;
    }
    
    public static Jama.Matrix square(Jama.Matrix matrix) {
        Jama.Matrix result = new Jama.Matrix(matrix.getRowDimension(), matrix.getColumnDimension());
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                result.set(i, j, Math.pow(matrix.get(i, j), 2));
            }
        }
        return result;
    }

    public static Jama.Matrix log(Jama.Matrix matrix) {
        Jama.Matrix result = new Jama.Matrix(matrix.getRowDimension(), matrix.getColumnDimension());
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                result.set(i, j, Math.log(matrix.get(i, j)));
            }
        }
        return result;
    }
    
    public static Jama.Matrix addConstant(Jama.Matrix matrix, double value) {
        double[][] oneArray = new double[matrix.getRowDimension()][matrix.getColumnDimension()];
        for (double[] oneVector : oneArray) {
            Arrays.fill(oneVector, value);
        }
        Jama.Matrix valueMatrix = new Jama.Matrix(oneArray);
        return matrix.plus(valueMatrix);
    }
    
    public static Jama.Matrix removeFirstColumn(Jama.Matrix matrix) {
        return matrix.getMatrix(0, matrix.getRowDimension()-1, 1, matrix.getColumnDimension()-1);
    }
    
    public static Jama.Matrix addOneColumn(Jama.Matrix x) {
        double[] srcRow = x.getRowPackedCopy();
        double[] dstRow = new double[x.getColumnDimension() + 1];
        System.arraycopy(srcRow, 0, dstRow, 1, srcRow.length);
        dstRow[0] = 1L;
        return new Jama.Matrix(dstRow, 1);
    }

    public static Jama.Matrix generateZ(int y, int k2) {
        double[][] z = new double[1][k2];
        for (int i = 0; i < k2; i++) {
            z[0][i] = y == i ? 1 : 0;
        }
        return new Jama.Matrix(z);
    }

}
