#include <bits/stdc++.h>
// #include <format>
#include <fstream>
#include <string>
#include <iostream>
using namespace std;

/*************************************************************************/
// https://github.com/MikeMirzayanov/testlib/blob/master/README.md


bool is_different(string output_file, string answer_file){
    ifstream output(output_file);
    ifstream answer(answer_file);
    string outputContent((istreambuf_iterator<char>(output)), istreambuf_iterator<char>());
    string answerContent((istreambuf_iterator<char>(answer)), istreambuf_iterator<char>());
    return outputContent != answerContent;
}

void rsystem(string command){
    system(command.c_str());
}

void csystem(string command){
    cout << command << endl;
    system(command.c_str());
}

int main (int argc, const char * argv[]){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    string name = argv[1];

    int n = 10; if(argc > 2) n = atoi(argv[2]);
    for(int i = 1; i <= n; i++){
        printf("i=%d\n", i);
        rsystem("./genInput.exe "+ to_string(i) +" > input.txt");
        rsystem("./"+ name +".exe < input.txt > output.txt");
        rsystem("./correct.exe < input.txt > answer.txt");
        rsystem("diff output.txt answer.txt");
        ifstream output("output.txt");
        ifstream answer("input.txt");

        string outputContent((istreambuf_iterator<char>(output)), istreambuf_iterator<char>());
        string answerContent((istreambuf_iterator<char>(answer)), istreambuf_iterator<char>());

        if (is_different("output.txt", "answer.txt")) {
            cout << "Different output for following input: " << endl;
            rsystem("cat input.txt");
            break;
        }
    }

    // Run random inputs
    // csystem("./random.sh " + name);
}
