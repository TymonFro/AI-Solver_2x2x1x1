#include<bits/stdc++.h>
//#define int long long
// #define pr pair<int,int>
// #define f first 
// #define s second 
#define F(I,K) for(int I = 0; I<K;I++)
#define F1(I,K) for(int I = 1; I<=K;I++)
using namespace std;

struct Cube2x2x1x1 {
    int corner_pos[4];
    int corner_orient[4];

    Cube2x2x1x1() {
        F(i,4) {
            corner_pos[i] = i;
            corner_orient[i] = 0;
        }
    }
};

map<string, int> move_map = {
    {"Rz2", 0}, {"Fx2", 1}, {"Rx", 2}, {"Rx'", 3}, {"Fz", 4}, {"Fz'", 5},
};
string moves_not[6] = {"Rz2", "Fx2", "Rx", "Rx'", "Fz", "Fz'",};

vector<vector<vector<int>>> MOVES = {
    // Rz2
    {{0,1,3,2}, {0,0,0,0}},
    // Fx2
    {{0,2,1,3}, {0,0,0,0}},
    // Rx
    {{0,1,2,3}, {0,0,1,1}},
    // Rx'
    {{0,1,2,3}, {0,0,-1,-1}},
    // Fz
    {{0,1,2,3}, {0,1,1,0}},
    // Fz'
    {{0,1,2,3}, {0,-1,-1,0}},
};

Cube2x2x1x1 applyMove(const Cube2x2x1x1& cube, int move) {
    Cube2x2x1x1 result;
    
    F(i,4){
        int src = MOVES[move][0][i];
        result.corner_pos[i] = cube.corner_pos[src];
        if((i+result.corner_pos[i]) % 2) result.corner_orient[i] = (cube.corner_orient[src] - MOVES[move][1][i] + 4) % 4;
        else result.corner_orient[i] = (cube.corner_orient[src] + MOVES[move][1][i] + 4) % 4;
    }
    
    return result;
}

bool isSolved(const Cube2x2x1x1& cube) {
    Cube2x2x1x1 solved = Cube2x2x1x1();
    F(i,4) if(cube.corner_pos[i] != solved.corner_pos[i] || cube.corner_orient[i] != solved.corner_orient[i]) return false;
    return true;
}

vector<double> encodeCube(const Cube2x2x1x1& cube){
    vector<double> input;
    F1(i,3){
        F(j,4){
            if(cube.corner_orient[i] == j) {
                input.push_back(1.0);
            } else {
                input.push_back(0.0);
            }
        }
    }
    F1(i,3){
        F(j,4){
            if(cube.corner_pos[i] == j) {
                input.push_back(1.0);
            } else {
                input.push_back(0.0);
            }
        }
    }
    return input;
}
