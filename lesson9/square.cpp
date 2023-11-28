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

vector<Point> generateRandomPointsInASquare(int n, double side) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-side/2., side/2.);

    vector<Point> points;
    for (int i = 0; i < n; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        points.push_back({x, y});
    }
    return points;
}

int main() {
    int n; // Number of points
    double side; // Radius of the circle

    cout << "Enter the number of points: ";
    cin >> n;

    cout << "Enter the side of the square: ";
    cin >> side;

    vector<Point> squarePoints = generateRandomPointsInASquare(n, side);

    ofstream outputFile("square.in");
    if (outputFile.is_open()) {
        for (const auto& point : squarePoints) {
            outputFile << point.x << " " << point.y << "\n";
        }
        outputFile.close();
        cout << "Points written to 'square.in'.\n";
    } else {
        cout << "Unable to open the file.\n";
    }

    return 0;
}