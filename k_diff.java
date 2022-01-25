import java.util.HashMap;
import java.util.Map;

//  Problem 2
//////////////////////////////////////////////////////
// // Author: Alexander Railton
// // Copyright: Copyright 2022, K-diff
// // Credits: Alexander Railton
// // Number: (416)-951-8621
// // Email: alexanderraiton@gmail.com
// // Status: {dev_status}
//////////////////////////////////////////////////////

public class k_diff 
{
    public static void main(String[] args) throws Exception 
    {
        int [] nums = new int []{3,1,4,1,5};
        int k = 2;
        System.out.println(findPairs(nums,k));
    }

    public static int findPairs(int[] nums, int k) 
    {
        if (nums == null || nums.length == 0 || k < 0)   return 0;
        
        // store each
        Map<Integer, Integer> map = new HashMap<>();
        int count = 0;
        for (int i : nums) 
        {
            map.put(i, map.getOrDefault(i, 0) + 1);
            System.out.println(map.put(i,map.getOrDefault(i,0)+1));
        }

        
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (k == 0) 
            {
                //count how many elements in the array that appear more than twice.
                if (entry.getValue() >= 2) {
                    count++;
                } 
            } 
            else {
                if (map.containsKey(entry.getKey() + k)) {
                    count++;
                }
            }
        }
        
        return count;
    }

}
