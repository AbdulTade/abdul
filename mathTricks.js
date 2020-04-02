var numQuestions;
// window.onload = () => {
    var numQuestions = Number(prompt("Enter the number questions you want to answer "));
// }

var operators = new Array(['+','-','*','/']);

function ndecimalPlaces(num,n){
    whole = Math.floor(num);
    rem = num- whole;
    dp = round(Math.pow(10,n) * rem) / Math.pow(10,n);
    return whole+dp;
}

var add = (num1,num2) => {
     return num1 + num2;
}

var subtract = (num1,num2) => {
    return num1 - num2;
}

var multiply = (num1,num2) => {
    return num1*num2;
}

var divide = (num1,num2) => {
    val = ndecimalPlaces(num1/num2,3);
    return val;
}



while(true){

    var i = 0;
    var score = 0;
    while(i < numQuestions){

       firstNum = Math.round(Math.random()*100);
       secondNum = Math.round(Math.random*100);
       opNum = round(Math.random()*3);
       num1Tag.innerText = String(firstNum);
       opNumTag.innerText = operators[opNum];
    opNumTag.innerText = '+';
       num2Tag.innerText = String(secondNum);

       button.addEventListener('click',() => {
        var num1Tag = document.getElementById('num1');
        var opNumTag = document.getElementById('operator');
        var num2Tag = document.getElementById('num2');
        var button = document.getElementById('button');
        var answer = document.getElementById('answer');

        firstNum = Math.round(Math.random()*100);
       secondNum = Math.round(Math.random*100);
       opNum = round(Math.random()*3);
       num1Tag.innerText = String(firstNum);
       opNumTag.innerText = operators[opNum];
    //    opNumTag.innerText = '+';
       num2Tag.innerText = String(secondNum);
        
           var solution = answer.value;
           funcJson = {

            '+' : add(firstNum+secondNum),
            '-' : subtract(firstNum,secondNum),
            '/' : divide(firstNum,secondNum),
            '*' : multiply(firstNum,secondNum)
 
        }
 
        var result = funcJson[operators[opNum]];

        // var result = add(firstNum,secondNum);

        
 
        if(solution === result){
            score++;
        }
 
        else {
            alert(`Incorrect. Correct answer: ${result}`);
             
        }
       i += 1;
 
     
    

       });// var operators = new Array(['+','-','*','/']);


       alert(`You scored ${score}/${numQuestions}`);
    var boolRes = confirm("Do you want to try again y/n ");
    if(boolRes == true){
        continue;
    }
    else {
        window.close();
        break;
    }

}
}