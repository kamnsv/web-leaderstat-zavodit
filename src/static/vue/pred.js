'use_strict';
var Predict = {
	
	props: ['classes', 'update'], 
		
	data() {
		return {
			show: false,
		}
	},//data
	
	
	
	components: {
		'dd-zone': DDZone,
	},//components
	
	methods: {
				
		close(){
			this.show = false;
		},//
		
		pred(){
			alert('predict')
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
						<dd-zone @loads="pred" ref="ddz"></dd-zone>  
					</div>
				</div>
			</div>
		</div>`
};//Classes  


