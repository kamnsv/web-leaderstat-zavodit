'use_strict';
var Classes = {

    props: ['classes'], 
	
	components: {
		'imgs': Imgs,
	},//components
	
	methods: {
		
		show(k){
			
			for (i in this.classes)
				if (i != k)
					this.classes[i].on = 0;
				
			if (k != undefined)
				this.classes[k].on = this.classes[k].on == 1 ? 0 : 1;
			
				
			

		}//show
		
	},//methods
    template:   `<div class="accordion w-100 classes"  id="classes">
					<div class="accordion-item classes__body" v-for="(v, k) in classes">
						<h2 class="accordion-header">
							
							<button :class="'accordion-button' +[' collapsed',''][v.on]" type="button" @click="show(k)">
							<span class="classes__nun">#{{k+1}}</span> 
							<span class="classes__name ms-1">{{v.name}}</span>
							</button>
						</h2>
						<div :class="'accordion-collapse collapse'+['',' show'][v.on]">
							<div class="accordion-body">
							<imgs :imgs="v.imgs" :name="v.name"></imgs>
							</div>
						</div>
					</div>
				</div>`
};//Classes  


