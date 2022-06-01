- Chạy web app:
```
python deploy.py
```

- API: Thêm tại Route **/extract**
	- Input: Normalize 4 góc file ảnh đã upload tại thư mục "**./uploads**", ảnh đã normalize sẽ save tại thư mục root "**./**".
	- Output: python object chứa dữ liệu đã extracted:
		- Properties:
			- status: "**OK**" nếu extract thành công, nếu status khác "**OK**" thì đây là failed to extract message để thông báo đến user.
			- Các properties còn lại là các mục thông tin được trích xuất
            ```python
            extracted_infos = {
                "status": "OK",
                "title": "info",
                "sub_title": "info2",
                "author": "info3",
                "date": "info4",
                "others": "info5",
                ...
            }
            ```