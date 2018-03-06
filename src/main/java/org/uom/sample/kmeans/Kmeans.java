package org.uom.sample.kmeans;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.stream.IntStream;

import mpi.MPI;
import mpi.MPIException;


public class Kmeans{

    public static void main(String[] args){

        if (args.length == 5){

            //reading the arguments
            int iterations = Integer.parseInt(args[0]);
            int k = Integer.parseInt(args[1]);
            int pointsCount = Integer.parseInt(args[2]);
            String centersFile = args[3];
            String pointsFile = args[4];

            int currentIteration = 0;

            try{

                MPI.Init(args);

                int rank = MPI.COMM_WORLD.getRank();
                int size = MPI.COMM_WORLD.getSize();

                int pointsPerThread = pointsCount/size;

                while (currentIteration < iterations){
                    double[] centers = readCenters(centersFile, k);
                    double[] points;

                    points = readPoints(pointsFile, rank * pointsPerThread, pointsPerThread);

                    ArrayList<ArrayList> pointCategories = new ArrayList();

                    for (int n = 0; n < k; n++){
                        pointCategories.add(new ArrayList<double[]>());
                    }

                    for (int i = 0; i < pointsPerThread; i++){
                        double minDistance = Double.MAX_VALUE;
                        int categoryBelonged = 0;

                        for (int m = 0; m < k; m++){
                            if (getEuclideanDistance(points, centers, 2*i, 2 * m) < minDistance){
                                minDistance = getEuclideanDistance(points, centers, 2*i, 2 * m);
                                categoryBelonged = m;
                            }
                        }

                        double[] currentPointSet = new double[2];
                        currentPointSet[0] = points[2*i];
                        currentPointSet[1] = points[2*i + 1];

                        pointCategories.get(categoryBelonged).add(currentPointSet);
                    }

                    double[] newCenters = new double[k*2];

                    for (int i = 0; i < pointCategories.size(); i++){
                        double xAve = 0.0;
                        double yAve = 0.0;

                        double tempSumX = 0;
                        double tempSumY = 0;

                        for (int m = 0; m < pointCategories.get(i).size(); m++){
                            double[] currentPointSet = (double[]) pointCategories.get(i).get(m);
                            tempSumX += currentPointSet[0];
                            tempSumY += currentPointSet[1];
                        }

                        MPI.COMM_WORLD.allReduce(tempSumX, xAve, 0, MPI.DOUBLE, MPI.SUM);
                        MPI.COMM_WORLD.allReduce(tempSumY, yAve, 0, MPI.DOUBLE, MPI.SUM);

                    }

                    if (rank==0){
                        writeCenters(centersFile, k, newCenters);

                        System.out.printf("Current iteration : %d\n", currentIteration);

                        writeIteration(currentIteration, rank);
                        currentIteration = readIteration(rank);

                        currentIteration++;
                    }

                }

                deleteIteration(rank);

                MPI.Finalize();

            } catch (IOException e){
                e.printStackTrace();
            } catch (MPIException e){
                e.printStackTrace();
            }

        } else {
            System.out.println("Incorrect number of parameters");
        }

    }

    private static void accumulate(double[] points, double[] centerSumsAndCounts, int pointOffset, int centerOffset, int dimension) {
        for (int i = 0; i < dimension; ++i) {
            centerSumsAndCounts[centerOffset+i] += points[pointOffset+i];
        }
    }

    private static double getEuclideanDistance(double[] point1, double[] point2, int point1Offset, int point2Offset) {
        double d = 0.0;
        for (int i = 0; i < 2; ++i) {
            d += Math.pow(point1[i+point1Offset] - point2[i+point2Offset], 2);
        }
        return Math.sqrt(d);
    }

    private static double[] readPoints(String pointsFile, int pointStartIdxForProc, int pointCountForProc) throws IOException {
        double[] points = new double[pointCountForProc*2];

        File file = new File(pointsFile);
        Scanner scanner = new Scanner(file);

        for (int i = 0; i < pointStartIdxForProc * 2; i++){
            scanner.nextDouble();
        }

        for (int i = 0; i < pointCountForProc * 2; i++){
            points[i] = scanner.nextDouble();
        }
        System.out.println(points.length);
        return points;
    }

    private static void resetCenterSumsAndCounts(double[] centerSumsAndCountsForThread) {
        IntStream.range(0, centerSumsAndCountsForThread.length).forEach(i -> centerSumsAndCountsForThread[i] = 0.0);
    }


    private static double[] readCenters(String centersFile, int k) throws IOException {
        double[] centers = new double[k*2];

        File file = new File(centersFile);
        Scanner scanner = new Scanner(file);

        for (int i = 0; i < k * 2; i++){
            centers[i] = scanner.nextDouble();
        }

        scanner.close();
        return centers;
    }

    private static void writeCenters(String centersFile, int k, double[] centers) throws IOException {
        FileWriter fileWriter = new FileWriter(centersFile);
        PrintWriter printWriter = new PrintWriter(fileWriter);

        for (int i = 0; i < k; i++){
            printWriter.printf("%.2f %.2f\n", centers[i*2], centers[i*2 + 1]);
        }
        printWriter.close();
    }

    private static int readIteration(int rank) throws IOException {
        int iteration;

        File file = new File("iterations" + rank +".txt");
        Scanner scanner = new Scanner(file);

        iteration = scanner.nextInt();
        scanner.close();

        return iteration;
    }

    private static void writeIteration(int iteration, int rank) throws IOException {
        FileWriter fileWriter = new FileWriter("iterations" + rank + ".txt");
        PrintWriter printWriter = new PrintWriter(fileWriter);
        printWriter.printf("%d", iteration);
        printWriter.close();
    }

    private static void deleteIteration(int rank){
        File file = new File("iterations" + rank + ".txt");
        file.delete();
    }

}