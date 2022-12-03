'use_strict';
var Classes = {

    props: ['classes'], 
	
	components: {
		'imgs': Imgs,
	},//components
	
	methods: {
		
		show(k){
			for (cls of this.classes)
				cls.on = 0;
			
			if (k!=undefined)
				this.classes[k].on = this.classes[k].on ? 0 : 1;
		}//show
		
	},//methods
    template:   `<div class="accordion w-100" id="classes">
					<div class="accordion-item" v-for="(v, k) in classes">
						<h2 class="accordion-header">
							<button :class="'accordion-button' +[' collapsed',''][v.on]" type="button" @click="show(k)">
							{{v.name}}
							</button>
						</h2>
						<div :class="'accordion-collapse collapse'+['',' show'][v.on]">
							<div class="accordion-body">
							<imgs :imgs="v.imgs"></imgs>
							</div>
						</div>
					</div>
				</div>`
};//Classes  


