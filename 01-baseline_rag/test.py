class Solution:
    def longestCommonPrefix(self, strs) -> str:
        s = ""
        for i in zip(*strs):
            print(i)
            if len(set(i)) == 1:
                s += i[0]
            else:
                break
        return s


if __name__ == '__main__':
    # test = Solution()
    # print(test.longestCommonPrefix(["flower","flow","flight"]))
    s = ["flower","flow","flight"]
    for i in zip(*s):
        print(i)