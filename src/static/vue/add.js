'use_strict';
var AddClass = {
	
	props: ['classes'], 
		
	data() {
		return {
			show: false,
			name: '',
			epoch: 10,
			count_load: 10,
			uload_inet: false,
			batch: 32,
			lr: 1000
		}
	},//data
	
	computed: {
			names(){
				let vals = new Set();
				for (cls of this.classes)
					vals.add(cls.name.toLowerCase())
				return vals;
			},
			
			learning_rate(){
				return this.lr/1000000;
			}
		
	},//computed
	
	
	components: {
		'dd-zone': DDZone,
	},//components
	
	methods: {
		add(){
			if (!this.$refs.ddz.valid) {
				return;
			}
			this.classes.push({name: this.name, on: 0})
			this.$refs.ddz.start_uload();
			this.show = false;
		},//add
		
		close(){
			this.show = false;
			
		}
		
	},//methods
    template:   `<button type="button" class="btn btn-outline-success show-add-form mt-2" @click="show=true">
		Добавить класс
		</button>
		
		
		<div class="modal d-block" v-if="show">
			<div class="modal-dialog">
				<div class="modal-content">
					<div class="modal-header">
						<h5 class="modal-title">Новый класс</h5>
						<button type="button" class="btn-close" @click="close()"></button>
					</div>
					<div class="modal-body">
					
					<div class="form-floating mb-3">
						<input v-model="name" type="text" class="form-control">
						<label for="floatingInput">Имя класса</label>
					</div>
					
					
					
					
					
					<label for="count-epochs" class="form-label">Количество эпох: {{epoch}}</label>
					<input v-model="epoch" type="range" class="form-range" min="5" max="100" id="count-epochs">
						
					<label for="batch-size" class="form-label">Размер партий: {{batch}}</label>
					<input v-model="batch" type="range" class="form-range" min="8" max="128" id="batch-size">

					<label for="learning-rate" class="form-label">Скорость обучения: {{learning_rate}}</label>
					<input v-model="lr" type="range" class="form-range" min="1" max="100000" id="learning-rate">

					<div class="input-group mb-3">
					<div class="input-group-text">
						<input v-model="uload_inet" class="form-check-input mt-0 me-2" type="checkbox" value="">
						Загрузить фото из интернета
					</div>
					<input type="number" v-model="count_load"  class="form-control" min="1" max="20">
					</div>


	


						
						
						<dd-zone :name="name" :classes="names" ref="ddz"></dd-zone>  
					</div>
					<div class="modal-footer">
						<button type="button" class="btn btn-secondary" @click="close()">Отмена</button>
						<button type="button" class="btn btn-primary" @click="add()">Сохранить</button>
					</div>
				</div>
			</div>
		</div>`
};//Classes  


