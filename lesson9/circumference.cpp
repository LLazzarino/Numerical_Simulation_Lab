#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

struct Point {
    double x;
    double y;
};

vector<Point> generateRandomPointsOnCircle(int n, double radius) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 2 * M_PI);

    vector<Point> points;
    for (int i = 0; i < n; ++i) {
        double angle = dis(gen);
        double x = radius * cos(angle);
        double y = radius * sin(angle);
        points.push_back({x, y});
    }
    return points;
}

int main() {
    int n; // Number of points
    double radius; // Radius of the circle

    cout << "Enter the number of points: ";
    cin >> n;

    cout << "Enter the radius of the circle: ";
    cin >> radius;

    vector<Point> circlePoints = generateRandomPointsOnCircle(n, radius);

    ofstream outputFile("circumference.in");
    if (outputFile.is_open()) {
        for (const auto& point : circlePoints) {
            outputFile << point.x << " " << point.y << "\n";
        }
        outputFile.close();
        cout << "Points written to 'circle_points.in'.\n";
    } else {
        cout << "Unable to open the file.\n";
    }

    return 0;
}