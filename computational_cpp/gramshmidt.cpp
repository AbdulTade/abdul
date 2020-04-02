#include <iostream>
#include <cmath>

using namespace std;

float orthonormalize(float Arr[3][3]);
float proj(float v[3],float u[3]);
float vectorDiff(float vec1[3],float vec2[3]);
typedef struct vectorData {
float vector[3];
};

int main(){

float vec[3][3] = {{1,1,1},{1,2,1},{2,2,2}};





return 0;
}

float proj(float v[3],float u[3]){

float dot;
float umag;

for( int i = 0; i < 3; i++){
   dot += v[i]*u[i];
}


for (int i = 0; i < 3; i++)
{
    umag += pow(u[i],2);
}

float k = dot/umag;

for (int i = 0; i < 3; i++)
{
    u[i] = k * u[i];
}

return *u;
}

void vectorDiff(float vec1[3],float vec2[3]){
    float vec3[3];
    for(int k = 0 ; k < 3; k++){
        
      vec3[k] = vec1[k] - vec2[k];

    }

    vectorData vecDiff;
    vecDiff.vector = vec3;

    return vecDiff

    
}

float orthonormalize(float Arr[3][3]){

float orth[3][3];

for(int i = 0; i < 3; i++){

    orth[i] = Arr[i] - 

}



}