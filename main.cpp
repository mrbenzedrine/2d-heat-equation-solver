#include <iostream>
#include <eigen3/Eigen/Eigen>

int main()
{
    
    int N_x = 4;
    int N_y = 4;
    double dx = 1.0/N_x;
    double dy = 1.0/N_y;
    double dt = 0.025;

    double startX = 0.0;
    double endX = 1.0;
    double startY = 0.0;
    double endY = 1.0;
    double startT = 0.0;
    double endT = 1.0/2.0;

    int noOfTimePoints = 0;
    double timeCounter = 0.0;

    while(timeCounter <= (endT - startT))
    {
        timeCounter += dt;
        noOfTimePoints++;
    }

    Eigen::VectorXd xSpacePoints, ySpacePoints, timePoints;
    xSpacePoints = Eigen::VectorXd::LinSpaced(N_x + 1, startX, endX);
    ySpacePoints = Eigen::VectorXd::LinSpaced(N_y + 1, startY, endY);
    timePoints = Eigen::VectorXd::LinSpaced(noOfTimePoints, startT, endT);

    Eigen::MatrixXd U(ySpacePoints.size(), xSpacePoints.size());
    U.setZero();

}
