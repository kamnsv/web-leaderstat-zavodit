'use_strict';
var DDZone = {
	
	props: ['name','classes'],
	
	data() {
		return {
			progon: false,
			vprog: 1,
			count: 0,
			arr: []
		}
	},//data
	
	computed: {
			valid(){

				if (this.name == '') return false;
				if (this.classes.has(this.name)) return false;
				return true;
			},
			
			progress(){
				return 100*this.vprog;
			},
		
	},//computed
	
	methods: {
		
		dragover(ev){
			ev.preventDefault();
		},//dragover
		
		drop(ev){
			ev.preventDefault();
			if (!this.valid) return;
			if (undefined == ev.dataTransfer) return;
			const typeit =  ev.dataTransfer.items ? true : false;  
			const files = typeit ? ev.dataTransfer.items : ev.dataTransfer.files;  
			this.count = files.length;
			for (var i = 0; i < files.length; i++){
				if (typeit) {                              
					if ("file" === files[i].kind)
					this.arr.push(files[i].getAsFile());                          
				} else {
					this.arr.push(files[i]);                             		
				}
			}

		},//drop
		
		load_files(ev){
			
			if (!this.valid) {
				ev.preventDefault();
				return;
			}
			
			ev.target.disabled = true;
			
			this.arr = [];
			this.count = ev.target.files.length;
			for (file of ev.target.files)
				this.arr.push(file);      
	    
			ev.target.disabled = false;	
			ev.target.value = null;
		},//load_files
		
	
		upload_file(file){
			if (!this.valid) return;

			new Promise((resolve, reject) => {
				let filename = file.name;
				let ext = filename.split('.').pop().toLowerCase();
				if (!['jpg','png'].includes(ext)){
					console.log(filename, 'file type is incorrect');	
					return;
				}
				let data = new FormData();
				let request = new XMLHttpRequest();
				
				let filesize = file.size;
				document.cookie = `filesize=${filesize}`;
				data.append("data", file);  
				data.append("name", this.name);  
				
				request.upload.addEventListener("progress", (e) => {
					let loaded = e.loaded;
					let total = e.total
					let percent_complete = loaded / total;
					console.log(filename, percent_complete);
					this.vprog = percent_complete;
					if (1 == percent_complete) 
						this.progon=false;
				});
				
				request.addEventListener("load", (e) => {
					if (request.status == 200) {
						console.log(filename, "load");
						this.progon=false;
						resolve();
					}
					else {
						console.log(filename, "bad");
						this.progon=false;
						reject(request.status);
					}
				});  
				
				request.addEventListener("error", (e) => {
					console.log(filename, "error");
					this.progon=false;
					reject();
					
				});  
				
				request.addEventListener("abort", (e) => {
					console.log(filename, "cansel");
					this.progon=false;
					resolve();
				});
		
				this.vprog = 0;
				this.progon = true;
				
				
				request.responseType = "json";
				request.open("post", `/put`);
				request.send(data);
			
			});
			


		},//upload_file
		
		open_upload(){
			if (this.valid)
				this.$refs.fileInput.click()
		},//open_upload
		
		
		start_uload(){
			
			for (var i = 0; i < this.arr.length; i++)                                 
				this.upload_file(this.arr[i]);
			
		},//start_uload
		
	},//methods
	
    template:   `<div :class="'dd ' + (valid ? '': 'disabled')">
					<div class="dd__zone"
					@dragover="dragover($event)"
					@drop="drop($event)"
					@click="open_upload()"
					>
						Загрузить картинки {{count ? '('+count+')' :  '' }}
					</div>
					
					<div class="progress" v-if="progon" >
						<div class="progress-bar" role="progressbar" :style="'width:' + progress +'%'" :aria-valuenow="progress" aria-valuemin="0" aria-valuemax="100"></div>
					</div>
					
					<input class="d-none" type="file" required="" multiple="" accept="image/jpeg,image/png,image/gif"
					@change="load_files($event)" ref="fileInput"
					/>
					
				</div>`
};//DDZone  