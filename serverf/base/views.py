from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
import numpy as np
import pandas as pd
from .apps import team_clustering
from .kmean import TeamClustering
import json
from .getMark import StudentAPI, AccountProcessor, JSONHandler
from pymongo import MongoClient
from .apps import team_clustering, final_df

test_student_tlu = []
data_path = 'D:\công việc\H\He thong kinh doanj thong minh\ServerTest\serverf\static\group_3_cleaned.csv'
login_url = "https://sinhvien1.tlu.edu.vn/education/oauth/token"
api_url = "https://sinhvien1.tlu.edu.vn/education/api/studentsubjectmark/getListStudentMarkBySemesterByLoginUser/0"
headers = {
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Cookie': 'SESSION=7a2b85c3-79cb-4527-bea7-d6d8f67e7300',
    'Origin': 'https://sinhvien1.tlu.edu.vn',
    'Referer': 'https://sinhvien1.tlu.edu.vn/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}
# team_clustering = None

# Create your views here.
def home(request):
    name = request.GET.get('name')

    response_data = {
        'name': name
    }
    
    # Trả về JSON response
    return JsonResponse(response_data)

def getMark2(username, password):
    subject_codes = ['CSE405', 'CSE404', 'CSE445']
    diff_subject_codes = ['CSE481', 'CSE486', 'CSE492']

    # Khởi tạo đối tượng API và JSON Handler
    student_api = StudentAPI(login_url, api_url, headers)
    json_handler = JSONHandler()

    # Khởi tạo đối tượng AccountProcessor và gọi hàm chính
    account_processor = AccountProcessor(student_api, json_handler)
    account_processor.process_accounts(username , password, subject_codes, diff_subject_codes)

    result_json = account_processor.get_result_json()

    return result_json

@csrf_exempt
def post_student_data(request):
    global checking_item, final_df
    if not request.session.get("students_data"):
        request.session["students_data"] = []

    if request.method == "POST":
        checking_item = None

        checking_item = None
        body = json.loads(request.body)
        marks = getMark2(body['msv'], body['password'])
        marks = json.loads(marks)
        mark_qthttt = None
        mark_data_mining= None
        mark_ML= None
        for mark in marks:
            if (mark['Subject Code'] == 'CSE405') or (mark['Subject Code'] == 'CSE481'):
                mark_qthttt = mark['Average Mark']
            elif (mark['Subject Code'] == 'CSE404') or (mark['Subject Code'] == 'CSE486'):
                mark_data_mining = mark['Average Mark']
            elif (mark['Subject Code'] == 'CSE445') or (mark['Subject Code'] == 'CSE492'):
                mark_ML = mark['Average Mark']

        student_data = [body['name'],mark_qthttt, mark_data_mining, mark_ML, body['hobby'], body['workingSkill'], body['mainActivity'], body['infoSource']]
        checking_item = team_clustering.add_student_and_predict(student_data)

        name = str(checking_item["Tên"])
        similarity_scores = list(map(float, checking_item["Độ tương thích"]))
        suitable_group = int(checking_item["Nhóm được phân"])
        highest_similarity_score = max(similarity_scores)

        data_response = {
            "name": name,
            "email": body['email'],
            "password": body['password'],
            "msv": body['msv'],
            "hobby": body['hobby'],
            "workingSkill": body['workingSkill'],
            "mainActivity": body['mainActivity'],
            "infoSource": body['infoSource'],
            "mark_qthttt": mark_qthttt,
            "mark_data_mining": mark_data_mining,
            "mark_ML": mark_ML,
            "similarity_scores": {
                "group_1": {
                    "similarity_score": similarity_scores[0]
                },
                "group_2": {
                    "similarity_score": similarity_scores[1]
                },
                "group_3": {
                    "similarity_score": similarity_scores[2]
                },
                "group_4": {
                    "similarity_score": similarity_scores[3]
                },
            },
            "suitable_group": suitable_group,
            "highest_similarity_score": highest_similarity_score
        }
        students_data = request.session["students_data"]
        students_data.append(data_response)
        request.session["students_data"] = students_data

        return JsonResponse({"message": "Data processed successfully.", "data": data_response}, status=201)
        
    elif request.method == "GET":
        students_data = request.session.get("students_data", [])
        if students_data:
            # Trả về phần tử đầu tiên của danh sách (bao gồm cluster và similarity_index)
            return JsonResponse(students_data[0], status=200)
        else:
            # Trả về thông báo nếu không có dữ liệu nào
            return JsonResponse({"error": "No data available"}, status=404)

    return JsonResponse({"error": "Method not allowed"}, status=405)