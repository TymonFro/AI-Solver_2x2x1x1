#include<bits/stdc++.h>
#include"2x2x1x1.h"
#include"net.h"
//#define int long long
#define pr pair<int,int>
#define f first 
#define s second 
#define F(I,K) for(int I = 0; I<K;I++)
#define F1(I,K) for(int I = 1; I<=K;I++)
using namespace std;
int scramble_moves = 1;

vector<vector<vector<double>>>h_w = {vector<vector<double>>(hidden_layers, vector<double>(in_layer)), vector<vector<double>>(hidden_layers, vector<double>(hidden_layers))};
vector<vector<double>>h_b(2, vector<double>(hidden_layers));
vector<vector<double>>o_w(out_layer, vector<double>(hidden_layers));
vector<double>o_b(out_layer);

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

void forwardpropagation(vector<double>& in_v, vector<vector<double>>&h_v, vector<double>&o_v){
    F(l,h_w.size()){
        F(i, h_w[l].size()){
            h_v[l][i] = h_b[l][i];
            if(l == 0){
                F(j, in_v.size()){
                    h_v[l][i] += in_v[j] * h_w[l][i][j];
                }
            }else{
                F(j, h_w[l][i].size()){
                    h_v[l][i] += h_v[l-1][j] * h_w[l][i][j];
                }
            }
            h_v[l][i] = relu(h_v[l][i]);
        }
    }
    int ll = h_w.size()-1;
    int ll_size = h_w[h_w.size()-1].size();

    F(i,o_w.size()){
        o_v[i] = o_b[i];
        F(j,ll_size){
            o_v[i] += h_v[ll][j] * o_w[i][j];
        }
        //o_v[i] = sigmoid(o_v[i]);
    }
}

signed main(){
    ifstream fin("2x2x1x1_net.txt");

    if(fin.good()){ //
        F(i,2){
            F(j,h_w[i].size()){
                F(k,h_w[i][j].size()){
                    fin >> h_w[i][j][k];
                }
            }
        }
        F(i,2){
            F(j,h_b[i].size()){
                fin >> h_b[i][j];
            }
        }
        F(i,o_w.size()){
            F(j,o_w[i].size()){
                fin >> o_w[i][j];
            }
        }
        F(i,o_b.size()){
            fin >> o_b[i];
        }
        fin.close();
    }else{
        cout<<"Network not generated!\n"; exit(0);
    }

    while(true){
        cout<<"Enter number of moves: ";
        cin >> scramble_moves;
        cout<<"Enter moves (Rz2, Fx2, Rx, Rx', Fz, Fz'): ";

        vector<int>moves;
        string x;
        F(i,scramble_moves){
            cin>>x; moves.push_back(move_map[x]);
        }

        Cube2x2x1x1 cube = Cube2x2x1x1();

        F(i,scramble_moves){
            cube = applyMove(cube, moves[i]);
        }

        for(int move_count = 0; ; move_count++){
            if(move_count >= scramble_moves+2){
                cout<<"Solution has not been found ;(\n"; break;
            }

            vector<double> in_v = encodeCube(cube);
            vector<vector<double>>h_v(2, vector<double>(hidden_layers, 0.0));
            vector<double>o_v(o_w.size(), 0.0);

            forwardpropagation(in_v, h_v, o_v);
            o_v = get_Q_values(o_v);

            int move = max_element(o_v.begin(), o_v.end()) - o_v.begin();
            //cout<<moves_not[move]<<" Q: "<<o_v[move]<<'\n';
            cout<<moves_not[move]<<' ';
            cube = applyMove(cube, move);

            if(isSolved(cube)){
                cout<<"Hypercube solved in "<<move_count+1<<" moves\n"<<endl;
                break;
            }
        }
    }
}