class vector3{

    constructor(arr){
        this.arr = arr;
        
    }
    
    dot(Arr){
        var dotP = 0;
        if(Arr.length !== 3){
            console.log(`Expected a vector of length ${3} got a vector of length ${Arr.length}`);
            return null;
        }
        for(var i = 0; i < 3 ; i++ ){
            dotP += this.arr[i] * Arr[i];
        }
        return dotP;
    }
 
    add(Arr){
        if(Arr.length !== 3){
            console.log(`Expected a vector of length ${3} got a vector of length ${Arr.length}`);
            return null;
        }
        var res = new Array();
        for(var j = 0; j < 3 ; j++){
            res[j] = this.arr[j] + Arr[j];
        }
        return res;
    }

    modulus(){
        var vecLen = 0;
        for(var j = 0; j < 3; j++){
            vecLen += Math.pow(this.arr[j],2); 
        }

        return Math.sqrt(vecLen);
    }

    // cross(Arr){
    //     var cres = new Array();

    // }

}

module.exports = vector3;