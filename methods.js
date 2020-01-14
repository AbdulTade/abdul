class method {
    
    constructor(){

    }

    ndecimalPlaces(num,n){

       var whole = Math.floor(num);
       var rem = num - whole;
       var dp = round(Math.pow(10,n)*rem) / Math.pow(10,n);
       return whole+dp

    }

    add(num1,num2){
        return num1+num2;
    }

    subtract(num1,num2){
        return num1 - num2;
    }

    multiply(num1,num2){
        return num1*num2;
    }

    divide(num1,num2){
        var val = this.ndecimalPlaces(num1/num2,3);
        return val;
    }


}

module.exports = method;