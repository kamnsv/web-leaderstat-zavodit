'use_strict';
var AddClass = {
	
	props: ['classes', 'update'], 
		
	data() {
		return {
			show: false,
			name: 'скейтборд',
			count_load: 10,
			load_inet: false,
			batch: 32,
			coef: 1
		}
	},//data
	
	computed: {
			names(){
				let vals = new Set();
				for (cls of this.classes)
					vals.add(cls.name.toLowerCase())
				return vals;
			},
			
		
	},//computed
	
	
	components: {
		'dd-zone': DDZone,
	},//components
	
	methods: {
		add(){
			if (!this.$refs.form.checkValidity()){
				this.$refs.form.reportValidity()
			} else {
				this.$refs.ddz.start_uload();
				this.$emit('update', this.name, this.coef, this.load_inet && this.count_load, this.batch);
				this.show = false;
			}
		},//add
		
		close(){
			this.show = false;
		}
		
	},//methods
    template:   `<button type="button" class="btn btn-outline-success show-add-form" @click="show=true">
		Добавить класс
		</button>
		
		
		<div class="modal d-block" v-if="show">
			<div class="modal-dialog">
				<div class="modal-content">
					<div class="modal-header">
						<h5 class="modal-title">Новый класс</h5>
						<button type="button" class="btn-close" @click="close()"></button>
					</div>
					<form class="modal-body" ref="form">
					
						<div class="form-floating mb-3">
							<input v-model="name" type="text" class="form-control" required name="name">
							<label for="floatingInput">Имя класса</label>
						</div>
						
							
						<label for="batch-size" class="form-label">Размер партий: {{batch}}</label>
						<input v-model="batch" type="range" class="form-range" min="8" max="128" id="batch-size">

						<label for="coef" class="form-label">Коэфициент аугментации: {{coef}}</label>
						<input v-model="coef" type="range" class="form-range" min="1" max="100" id="coef">

						<div class="input-group mb-3">
						<div class="input-group-text">
							<input v-model="load_inet" class="form-check-input mt-0 me-2" type="checkbox" value="">
							Загрузить фото из интернета
						</div>
						<input type="number" v-model="count_load"  class="form-control" min="1" max="100">
						</div>


							
							<dd-zone :name="name" :classes="names" route="put" ref="ddz"></dd-zone>  
					</form>
					<div class="modal-footer">
						<button type="button" class="btn btn-secondary" @click="close()">Отмена</button>
						<button type="button" class="btn btn-success" @click="add()">Добавить</button>
					</div>
				</div>
			</div>
		</div>`
};//Classes  


