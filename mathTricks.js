var numQuestions;
window.onload = () => {
    var numQuestions = parseInt(prompt("Enter the number questions you want to answer "));
}
var num1Tag = document.getElementById('num1');
var opNumTag = document.getElementById('operator');
var num2Tag = document.getElementById('num2');
var button = document.getElementById('button');
var answer = document.getElementById('answer');

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
       firstNum = round(Math.random()*100);
       secondNum = round(Math.random*100);
       opNum = round(Math.random()*3);
       num1Tag.innerHTML = String(firstNum);
       opNumTag.innerHTML = operators[opNum];
       num2Tag.innerHTML = String(secondNum);

       button.addEventListener('onclick',() => {
           var solution = answer.value;
           funcJson = {

            '+' : add(firstNum+secondNum),
            '-' : subtract(firstNum,secondNum),
            '/' : divide(firstNum,secondNum),
            '*' : multiply(firstNum,secondNum)
 
        }
 
        var result = funcJson[operators[opNum]];
 
        if(solution === result){
            score++;
        }
 
        else {
            alert(`Incorrect. Correct answer: ${result}`);
 
        }
       i += 1;
 
     
    alert(`You scored ${score}/${numQuestions}`);
    var boolRes = confirm("Do you want to try again y/n ");
    if(boolRes == true){
        continue;
    }
    else {
        boolRes == false;
        window.close();
    }
    
       });

       
}
}