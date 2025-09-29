#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include "Node.h"

using namespace std;

int main() {
    cout << "Estas dentro de una simulación\n";
    const int N=10;
    std::vector<Node> nodes_;
    nodes_.reserve(N);


    for (int i = 0; i<N; ++i ){
        nodes_.emplace_back(i);
        nodes_[i].set(static_cast<double>(i) * 3);
    }
    cout << "Los nodos secretos para salir de la matrix son: \n";

    for (int i = 0; i<N; ++i ){
        cout << "Nodo id:  " << i << " cuyo poder es = " << nodes_[i].get() << " \n" ; 
    }

    return 0;
}


