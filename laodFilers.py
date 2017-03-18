#!/usr/bin/python
import os
import argparse


def loadFileDicts () :
     fileList = []
     
     args = argparse.ArgumentParser(prog='test', description='a simple text processor')
     args.add_argument('--folder',required=True,type=str,help="provide a working dir")
     
     parser = args.parse_args()

     if os.path.exists(parser.folder) :
          fileList = os.listdir(parser.folder)
     
     return fileList

if __name__ == '__main__':
     
     loadFileDicts ()
     
     
