// ECS629/759 Assignment 2 - ID3 Skeleton Code
// Author: Simon Dixon

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;
import java.util.Arrays;
import java.util.ArrayList;

class ID3 {

	class TreeNode {

		TreeNode[] children;
		int value;

		public TreeNode(TreeNode[] ch, int val) {
			value = val;
			children = ch;
		} // constructor

		public String toString() {
			return toString("");
		} // toString()

		String toString(String indent) {
			if (children != null) {
				String s = "";
				for (int i = 0; i < children.length; i++)
					s += indent + data[0][value] + "=" +
							strings[value][i] + "\n" +
							children[i].toString(indent + '\t');
				return s;
			} else
				return indent + "Class: " + strings[attributes-1][value] + "\n";
		} // toString(String)

	} // inner class TreeNode

	private int attributes; 	// Number of attributes (including the class)
	private int examples;		// Number of training examples
	private TreeNode decisionTree;	// TreeNode learnt in training, used for classifying
	private String[][] data;	// Training data indexed by example, attribute
	private String[][] strings; // Unique strings for each attribute
	private int[] stringCount;  // Number of unique strings for each attribute

	public ID3() {
		attributes = 0;
		examples = 0;
		decisionTree = null;
		data = null;
		strings = null;
		stringCount = null;
	} // constructor

	public void printTreeNode() {
		if (decisionTree == null)
			error("Attempted to print null TreeNode");
		else
			System.out.println(decisionTree);
	} // printTreeNode()

	/** Print error message and exit. **/
	static void error(String msg) {
		System.err.println("Error: " + msg);
		System.exit(1);
	} // error()

	static final double LOG2 = Math.log(2.0);

	static double xlogx(double x) {
		return x == 0? 0: x * Math.log(x) / LOG2;
	} // xlogx()

        //AUTHOR: Alex Drossos
	/** Execute the decision tree on the given examples in testData, and print
	 *  the resulting class names, one to a line, for each example in testData.
	 **/
	public void classify(String[][] testData) {
		if (decisionTree == null)
			error("Please run training phase before classification");
		// PUT  YOUR CODE HERE FOR CLASSIFICATION
                int length = testData.length;
                //increment through all the data given as an input
		for(int dataInc = 1; dataInc < length; dataInc++){
                        //classifier only takes a one dimensional String array as an input 
                        //and takes in the whole decision tree that was created inside 
                        //the train function
			int result = classifier(testData[dataInc], decisionTree);
                        //once the classifier function has gone all the way through the tree
                        //it prints out the resulting attribute string where the value is 
                        //stored in result
                        String toPrint = strings[attributes-1][result];
			System.out.println(toPrint);
		}
	} // classify()

	private int classifier(String[] testData, TreeNode current){
		//where recursion stops because it has reached a class node
                //AKA no children in the node data structure
                if(current.children == null){
			return current.value;
		}
                //this again is where the recursion comes in, it contintues to classify 
                //through the whole decisionTree
                //loops through the amount of attributes in the current treeNode
		for(int i = 0; i < stringCount[current.value]; i++){
			if(strings[current.value][i].equals(testData[current.value])){
                            //continue to recurse when the branch string equal
                            //to the current data value
                            return classifier(testData, current.children[i]);
			}
		}
		return 0;
	}

	public void train(String[][] trainingData) {
		indexStrings(trainingData);
		// PUT  YOUR CODE HERE FOR TRAINING
                //create empty used array
		ArrayList<Integer> allCols = new ArrayList<Integer>();
                ArrayList<Integer> allRows = new ArrayList<Integer>();
		decisionTree = ID3new(allCols, allRows);

	} // train()

	private TreeNode ID3new(ArrayList<Integer> usedAtt, ArrayList<Integer> split){
		//Check if all the classes in the remaining set are equal
                int val = checkAllSame(split);
		if(val != -1){
                    //If value of class is anything besides -1, return that leaf node
			return new TreeNode(null, val);
		}
                int attLeft = attributes - usedAtt.size();
                int actualSize = examples - 1;
                //Create new arrayList to update the set 
                ArrayList<Integer> updateAtt = new ArrayList<Integer>(usedAtt);
		//If no attributes left, return majority class
                if(attLeft == 0){
                    return new TreeNode(null, getMajClass(split));
		}
                //If there are no more examples to go through..
                if(split.size() == actualSize){
			return null;
		}
                //calculate using the the method and add the resulting attrib val to the arrayList
		int[][][] cData = retClassData(usedAtt, split);
		double bestGain = -1;
		int bestGainAtt = -1;
                //FIND BEST QUESTION by looping through the remaining attributes
		for(int col = 0; col < cData.length; col++){
			//check if attribute has been used
			int checkVal = alreadySplit(col,usedAtt);
                        if(checkVal == -1){
                            continue;
                        }
			double tempGain = 0;
                        //count up total number of strings in that set of data
                        //this for loop means that 
                        for(int[] str : cData[col]){
				//adding up each string in that attribute
				int tot = 0; 
                                double entropy = 0;
                                for (int i = 0; i < str.length; i++){
                                    tot+= str[i];
                                }
                                for (int j = 0; j < str.length; j++){
                                    double prob = ((double)str[j])/((double)tot);
                                    entropy -= xlogx(prob);
                                }
                                int rowsLeft = examples - 1 - split.size();
				double setEnt = ((double)tot/(double)rowsLeft);
				tempGain -= setEnt * entropy;
			}
                        //continuously compare per attribute, and replace if temp 
                        //is higher than current gain 
			if(tempGain > bestGain){
				bestGain = tempGain;
				bestGainAtt = col;
			}
		}
		updateAtt.add(bestGainAtt);
                int childCount = stringCount[bestGainAtt];
		//create array of TreeNodes for the children of the new question
		TreeNode[] newNodes = new TreeNode[childCount];
                int iteration = newNodes.length;
		for(int child = 0; child < iteration; child++){
			//Each iteration creates new set and a child based on that set
                        ArrayList<Integer> newSplit = shrinkSet(bestGainAtt, child, split);
			newNodes[child] = ID3new(updateAtt, newSplit);
                        //if the resulting child is null return majority class
			if(newNodes[child] == null){
				newNodes[child] = new TreeNode(null, getMajClass(split));
			}
		}
		return new TreeNode(newNodes, bestGainAtt);
	}
        
        private int checkAllSame(ArrayList<Integer> split){
            int val = -1;
		for(int i = 1; i < examples; i++){
			int checkVal = alreadySplit(i,split);
                        if(checkVal == -1){
                            continue;
                        }
			String string = data[i][attributes-1];
                        int stop = stringCount[attributes-1];
			//loop through each element in the class strings and find 
			for(int it = 0; it < stop; it++){
                                String compare = strings[attributes-1][it];
				if(string.equals(compare)){
					val = it;
					break;
				}
			}
		}
		//loop through each row to see if they're the same values 
		for(int r = 1; r < examples; r++){
			int checkVal = alreadySplit(r,split);
                        if(checkVal == -1){
                            continue;
                        }
                        //If one data value is not equal to the class, then it's automatically
                        //not the same.
                        //Value of -1 should be returned because the indexes of class start at 0
			if(!strings[attributes-1][val].equals(data[r][attributes-1])){
				return -1;
			}
		}
		//return class that is same for all rows;
		return val;
        }      

	private int getMajClass(ArrayList<Integer> split){
                int numOfClasses = stringCount[attributes-1];
		//this creates an int array of the number of each class
                //corresponding to its index
                int[] clsNums = new int[numOfClasses];
		for(int i = 1; i < examples; i++){
                    int checkVal = alreadySplit(i,split);
                    if(checkVal == -1){
                        continue;
                    }
                    //loop through all the classes to see which index the data belongs to
                    for(int num = 0; num <clsNums.length; num++){
                        //finding class index by checkin if the data on that row matches the 
                        //class with the value num
                        if(data[i][attributes-1].equals(strings[attributes-1][num])){
                            //updates the value in that class index 
                            clsNums[num]++;
			}
                    }
		}
		//find highest number of all array
                //this is comparing each element with its neighbor
                //first iteration would be"  if clasNum[1] > clasNums[0]
		int largest = 0;
		for(int j = 1; j < clsNums.length + 1; j++){
			if(clsNums[j] > clsNums[largest]){
                            //if it's bigger, than that INDEX is the class number
                            //NOT THE VALUE INSIDE that index
				largest = j;
			}
		}
		return largest;
        }

	private ArrayList<Integer> shrinkSet(int bestGainAtt, int it, ArrayList<Integer> split){
                ArrayList<Integer> newSplit = new ArrayList<Integer>(split);
		if(strings[bestGainAtt][it] == null){
                    return split;
                }
		for(int i = 1; i < examples; i++){
			//if string is not equal cell value it is added to ignored
			//but if it is already ignored, there is no need for it to be there twice
                        int checkVal = alreadySplit(i,split);
			if ((!strings[bestGainAtt][it].equals(data[i][bestGainAtt])) && (checkVal != -1)){
                            newSplit.add(i);
                        }  
		}
		return newSplit;
	}

	private int alreadySplit(int val, ArrayList<Integer> ignore){
            //if the value returned is -1, then it's been used
            int returnVal = 0;
            if (ignore.stream().anyMatch((ignored) -> (val == ignored))) {
                returnVal = -1;
            }
		return returnVal;
	}

        
        //this function will return an integer array that will be used to loop through the 
        //remaining attributes and then sets to calculate each individual entropy
	private int[][][] retClassData(ArrayList<Integer> usedAtt, ArrayList<Integer> split){
                int[][][] cntr = new int[attributes-1][][];
                int classAmount = stringCount[attributes-1];
                //loop through the columns
		for(int col = 0; col < attributes - 1; col++){
			//as always, check if the attribute vlaue is already being ignored
                        int checkVal = alreadySplit(col,usedAtt);
                        if(checkVal == -1){
                            continue;
                        }
                        //create an array for the first value of the cntr array
			cntr[col] = new int[stringCount[col]][classAmount];
                        //loop through each row
			for(int r = 1; r < examples; r++){
                                int stringVal = 0;
                                //check if the row has already been split
				int check = alreadySplit(r,split);
                                    if(check == -1){
                                        continue;
                                }
                                //this for loop says for every value of strings[col] make 
                                //"num" a place holder to compare values inside the for loop
                                for(String num : strings[col]){
                                        int classVal = 0;
					//find the string index using this for loop
                                        if(data[r][col].equals(num)){
                                                //looping through the array of class values in strings
						for(String c : strings[attributes-1]){
                                                        //check which class name 
                                                        //the data on the current row iteration equals
							if(data[r][attributes-1].equals(c)){
                                                            //increment the value of these particular indexes
                                                            cntr[col][stringVal][classVal]++;
							}
                                                        if(c == null){
                                                            continue;
                                                        }
                                                        //increment classVal throughout this for loop and this its 
                                                        //reset when num increments in ITS for loop
							classVal = classVal + 1;
						}
					}
                                        if(num == null){
                                            continue;
                                        }
					stringVal = stringVal + 1;
				}
			}
		}
		return cntr;
	}

	/** Given a 2-dimensional array containing the training data, numbers each
	 *  unique value that each attribute has, and stores these Strings in
	 *  instance variables; for example, for attribute 2, its first value
	 *  would be stored in strings[2][0], its second value in strings[2][1],
	 *  and so on; and the number of different values in stringCount[2].
	 **/
	void indexStrings(String[][] inputData) {
		data = inputData;
		examples = data.length;
		attributes = data[0].length;
		stringCount = new int[attributes];
		strings = new String[attributes][examples];// might not need all columns
		int index = 0;
		for (int attr = 0; attr < attributes; attr++) {
			stringCount[attr] = 0;
			for (int ex = 1; ex < examples; ex++) {
				for (index = 0; index < stringCount[attr]; index++)
					if (data[ex][attr].equals(strings[attr][index]))
						break;	// we've seen this String before
				if (index == stringCount[attr])		// if new String found
					strings[attr][stringCount[attr]++] = data[ex][attr];
			} // for each example
		} // for each attribute
	} // indexStrings()

	/** For debugging: prints the list of attribute values for each attribute
	 *  and their index values.
	 **/
	void printStrings() {
		for (int attr = 0; attr < attributes; attr++)
			for (int index = 0; index < stringCount[attr]; index++)
				System.out.println(data[0][attr] + " value " + index +
									" = " + strings[attr][index]);
	} // printStrings()

	/** Reads a text file containing a fixed number of comma-separated values
	 *  on each line, and returns a two dimensional array of these values,
	 *  indexed by line number and position in line.
	 **/
	static String[][] parseCSV(String fileName)
								throws FileNotFoundException, IOException {
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String s = br.readLine();
		int fields = 1;
		int index = 0;
		while ((index = s.indexOf(',', index) + 1) > 0)
			fields++;
		int lines = 1;
		while (br.readLine() != null)
			lines++;
		br.close();
		String[][] data = new String[lines][fields];
		Scanner sc = new Scanner(new File(fileName));
		sc.useDelimiter("[,\n]");
		for (int l = 0; l < lines; l++)
			for (int f = 0; f < fields; f++)
				if (sc.hasNext())
					data[l][f] = sc.next();
				else
					error("Scan error in " + fileName + " at " + l + ":" + f);
		sc.close();
		return data;
	} // parseCSV()

	public static void main(String[] args) throws FileNotFoundException,
												  IOException {
		if (args.length != 2)
			error("Expected 2 arguments: file names of training and test data");
		String[][] trainingData = parseCSV(args[0]);
		String[][] testData = parseCSV(args[1]);
		ID3 classifier = new ID3();
		classifier.train(trainingData);
		classifier.printTreeNode();
		classifier.classify(testData);
	} // main()

} // class ID3
