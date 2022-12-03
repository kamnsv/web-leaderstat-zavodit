'use_strict';
var Predict = {
	
	props: ['result'], 
		
	data() {
		return {
			show: false,
			predicting: false
		}
	},//data
	
	
	
	components: {
		'dd-zone': DDZone,
	},//components
	
	methods: {
				
		close(){
			this.show = false;
		},//
		
		load_files(){
			this.predicting = true;
			let name = (new Date()).toISOString().replaceAll(':','-')
			this.$refs.ddz.start_uload(name);
			this.$emit('pred', name);
		}
		
	},//methods
    template:   `<button type="button" class="btn btn-outline-primary show-add-form" @click="show=true">
		Распознать
		</button>
		
		<div class="modal d-block" v-if="show">
			<div class="modal-dialog">
				<div class="modal-content">
					<div class="modal-header">
						<h5 class="modal-title">Классификация изображений</h5>
						<button type="button" class="btn-close" @click="close()"></button>
					</div>
					<div class="modal-body">
						<dd-zone @loads="load_files" route="tmp" ref="ddz"></dd-zone>  
					</div>
					
					<ul class="list-group list-group-flush" v-for="i in result">
					<li class="list-group-item">
							<div class="d-flex">
								<img :src="i.src" class="img-thumbnail" style="width: 200px"/>
								<ol class="list-group list-group-numbered">
									<li v-for="j in i.res" class="list-group-item">{{j}}</li>
								</ol>
	
							</div>	
						</li>
					</ul>
					
					
					<div class="spinner-border text-primary mb-2 mx-auto" role="status" v-if="predicting">
						<span class="visually-hidden">Loading...</span>
					</div>
					
					<div class="modal-footer">
						<button type="button" class="btn btn-primary" @click="close()" v-if="result.length">Закрыть</button>
					</div>

				</div>
			</div>
		</div>`
};//Predict  


