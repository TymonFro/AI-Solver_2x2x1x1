#include<bits/stdc++.h>
#include"2x2x1x1.h"
//#define int long long
#define pr pair<int,int>
#define f first 
#define s second 
#define F(I,K) for(int I = 0; I<K;I++)
#define F1(I,K) for(int I = 1; I<=K;I++)
using namespace std;
int scramble_moves;

signed main(){
    Cube2x2x1x1 cube = Cube2x2x1x1();
    while(true){
        cout<<"Enter moves (0-7): ";
        int move;
        cin >> move;
        
        cube = applyMove(cube, move);

        F(i,4){
            cout << "Position " << i << ": Piece = " << cube.corner_pos[i] << ", Orientation = " << cube.corner_orient[i] << endl;
        }
        

        if(isSolved(cube)){
            cout<<"Hypercube solved!\n";
        }else{
            cout<<"Hypercube not solved.\n"; 
        }
    }
}