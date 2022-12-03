'use_strict';
var root = {
	data() 
	{
	return {
			classes: [],
			flash:[]
		}		
	},//data
	
	computed: {
		
		
	},//computed
	
	methods: {
		add_class(name, epoch, count, batch, lr){
			
				data = {
					name: name,
					batch: batch,
					epoch: epoch,
					lr: lr, 
					imgs: [], 
					coef: 1,  
					load: count === false? 0 : count,  
				}
				if (null == this.ws)
					this.ini_ws();
				this.ws.send(JSON.stringify(data));

			
		},//add_class
		

	    load_classes(){
		  
			(async () => {
				const rawResponse = await fetch('/classes', {
					method: 'GET',
				});
				try{
					this.classes = await rawResponse.json();
					for (cls of this.classes)
						cls.on = 0;
				} catch(e){
					this.flash.push('Ошибка сервиса api /classes')
				}
			})();
		
		},//load_classes
			
	  	ini_ws(){
			let url = `ws://${location.hostname}:${location.port}/ws`;
			this.ws = new WebSocket(url);
			this.ws.onopen = (e) => 
			{
				console.log('ws open');
			}
				
			this.ws.onclose = (e) => 
			{
				this.ws = null;
			}
			this.ws.onmessage = (e) =>
			{
				console.log(e.data);
			}
			this.ws.onerror = (e) =>
			{
				console.log(e);
				this.ws.close();
			}
		},//ini_ws

	},//methods
		
	mounted() {
		this.load_classes();
		this.ini_ws();
		
	},//mounted
	
	components: {
		'classes-box': Classes,
		'add-class': AddClass,
		'predict-box': Predict
	},//components
	
}//root

const app = Vue.createApp(root);
const vm = app.mount('#app');