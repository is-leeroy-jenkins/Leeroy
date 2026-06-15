/*
 * ==========================================================================================
 *  Leeroy Documentation JavaScript
 *  File: docs/assets/js/leeroy.js
 *
 *  Purpose:
 *      Provides safe progressive enhancements for the Leeroy MkDocs Material site.
 *      This script is dependency-free, fails gracefully when optional page elements are
 *      absent, and is designed for GitHub Pages, Material for MkDocs, and mkdocstrings.
 *
 *  Features:
 *      - Reading progress bar
 *      - Scroll-to-top button
 *      - Page tools: copy page link and print page
 *      - Heading anchor copy buttons
 *      - API tools panel with filter, expand all, collapse all, and clear filter
 *      - mkdocstrings API member badges
 *      - Large-table filtering
 *      - Code-block language labels
 *      - Code-block expand/collapse for long examples
 *      - External-link hardening
 *      - Search placeholder customization
 *      - Active navigation scroll memory
 *      - Page path metadata
 *      - Keyboard focus mode
 *      - Active table-of-contents highlighting
 *
 *  Compatibility:
 *      - MkDocs Material
 *      - mkdocstrings
 *      - GitHub Pages
 *      - Modern Chromium, Edge, Firefox, Safari
 *
 *  Notes:
 *      This file intentionally avoids external dependencies, network calls, analytics,
 *      cookies, and local-storage values containing user content.
 * ==========================================================================================
 */
( function()
{
	"use strict";
	const LeeroyDocs = {
		config: {
			siteName: "Leeroy",
			scrollTopId: "leeroy-scroll-top",
			pageToolsId: "leeroy-page-tools",
			progressId: "leeroy-reading-progress",
			apiSearchId: "leeroy-api-search",
			tableFilterClass: "leeroy-table-filter",
			headingLinkClass: "leeroy-heading-link",
			codeLabelClass: "leeroy-code-label",
			codeToggleClass: "leeroy-code-toggle",
			initializedAttribute: "data-leeroy-enhanced",
			navScrollKey: "leeroy-docs-nav-scroll",
			maxCollapsedCodeHeight: 420,
			largeTableMinimumRows: 8,
			headingSelector: ".md-typeset h2[id], .md-typeset h3[id], .md-typeset h4[id]",
			contentSelector: ".md-content__inner",
			navSelector: ".md-nav--primary .md-nav__list",
			tocSelector: ".md-nav--secondary",
			tableSelector: ".md-typeset table:not([data-leeroy-no-filter])",
			codeSelector: ".md-typeset pre > code",
			apiObjectSelector:
					".doc.doc-object, .doc-class, .doc-function, .doc-method, .doc-attribute, .doc-property"
		},
		state: {
			scrollTicking: false,
			resizeTicking: false
		},
		init: function()
		{
			if( document.documentElement.getAttribute( this.config.initializedAttribute ) ===
					"true" )
			{
				return;
			}
			document.documentElement.setAttribute( this.config.initializedAttribute, "true" );
			this.enhanceExternalLinks();
			this.customizeSearch();
			this.addReadingProgress();
			this.addScrollTopButton();
			this.addPageTools();
			this.addHeadingLinks();
			this.addTableFilters();
			this.addCodeLabels();
			this.addCodeToggles();
			this.addPagePathMetadata();
			this.restoreNavigationScroll();
			this.enhanceKeyboardFocus();
			this.enhanceApiReference();
			this.addApiTools();
			this.enhanceTocProgress();
			this.addMermaidGuard();
			this.bindLifecycleEvents();
			this.updateReadingProgress();
			this.updateScrollTopVisibility();
			this.updateTocProgress();
		},
		bindLifecycleEvents: function()
		{
			const self = this;
			window.addEventListener( "scroll", function()
			{
				if( !self.state.scrollTicking )
				{
					window.requestAnimationFrame( function()
					{
						self.updateReadingProgress();
						self.updateScrollTopVisibility();
						self.updateTocProgress();
						self.state.scrollTicking = false;
					} );
					self.state.scrollTicking = true;
				}
			}, { passive: true } );
			window.addEventListener( "resize", function()
			{
				if( !self.state.resizeTicking )
				{
					window.requestAnimationFrame( function()
					{
						self.updateReadingProgress();
						self.updateTocProgress();
						self.state.resizeTicking = false;
					} );
					self.state.resizeTicking = true;
				}
			}, { passive: true } );
			document.addEventListener( "click", function( event )
			{
				self.handleDocumentClick( event );
			} );
			document.addEventListener( "keydown", function( event )
			{
				self.handleKeyboardShortcuts( event );
			} );
			window.addEventListener( "beforeunload", function()
			{
				self.saveNavigationScroll();
			} );
			if( typeof document$ !== "undefined" && document$ && typeof document$.subscribe ===
					"function" )
			{
				document$.subscribe( function()
				{
					document.documentElement.removeAttribute( self.config.initializedAttribute );
					setTimeout( function()
					{
						self.init();
					}, 25 );
				} );
			}
		},
		handleDocumentClick: function( event )
		{
			const target = event.target;
			if( !target || !target.closest )
			{
				return;
			}
			if( target.closest( "#" + this.config.scrollTopId ) )
			{
				event.preventDefault();
				this.scrollToTop();
				return;
			}
			if( target.closest( "[data-leeroy-copy-heading]" ) )
			{
				event.preventDefault();
				this.copyHeadingLink( target.closest( "[data-leeroy-copy-heading]" ) );
				return;
			}
			if( target.closest( "[data-leeroy-copy-page]" ) )
			{
				event.preventDefault();
				this.copyPageLink( target.closest( "[data-leeroy-copy-page]" ) );
				return;
			}
			if( target.closest( "[data-leeroy-print-page]" ) )
			{
				event.preventDefault();
				window.print();
				return;
			}
			if( target.closest( "[data-leeroy-toggle-code]" ) )
			{
				event.preventDefault();
				this.toggleCodeBlock( target.closest( "[data-leeroy-toggle-code]" ) );
				return;
			}
			if( target.closest( "[data-leeroy-api-expand]" ) )
			{
				event.preventDefault();
				this.setApiDetailsState( true );
				return;
			}
			if( target.closest( "[data-leeroy-api-collapse]" ) )
			{
				event.preventDefault();
				this.setApiDetailsState( false );
				return;
			}
			if( target.closest( "[data-leeroy-api-clear]" ) )
			{
				event.preventDefault();
				this.clearApiFilter();
			}
		},
		handleKeyboardShortcuts: function( event )
		{
			const key = ( event.key || "" ).toLowerCase();
			if( event.altKey && key === "t" )
			{
				event.preventDefault();
				this.scrollToTop();
				return;
			}
			if( event.altKey && key === "p" )
			{
				event.preventDefault();
				window.print();
				return;
			}
			if( event.altKey && key === "l" )
			{
				event.preventDefault();
				this.copyCurrentPageToClipboard();
				return;
			}
			if( event.altKey && key === "f" )
			{
				const apiSearch = document.getElementById( this.config.apiSearchId );
				if( apiSearch )
				{
					event.preventDefault();
					apiSearch.focus();
				}
			}
		},
		enhanceExternalLinks: function()
		{
			const links = document.querySelectorAll( ".md-typeset a[href]" );
			const currentHost = window.location.host;
			links.forEach( function( link )
			{
				try
				{
					const url = new URL( link.href, window.location.href );
					if( url.host && url.host !== currentHost )
					{
						link.setAttribute( "target", "_blank" );
						link.setAttribute( "rel", "noopener noreferrer" );
						link.classList.add( "leeroy-external-link" );
						if( !link.querySelector( ".leeroy-external-indicator" ) )
						{
							const indicator = document.createElement( "span" );
							indicator.className = "leeroy-external-indicator";
							indicator.setAttribute( "aria-hidden", "true" );
							indicator.textContent = " ↗";
							link.appendChild( indicator );
						}
					}
				}
				catch( error )
				{
					return;
				}
			} );
		},
		customizeSearch: function()
		{
			const searchInputs = document.querySelectorAll( "input.md-search__input" );
			searchInputs.forEach( function( input )
			{
				input.setAttribute( "placeholder", "Search Leeroy docs..." );
				input.setAttribute( "aria-label", "Search Leeroy documentation" );
			} );
		},
		addReadingProgress: function()
		{
			if( document.getElementById( this.config.progressId ) )
			{
				return;
			}
			const progress = document.createElement( "div" );
			progress.id = this.config.progressId;
			progress.setAttribute( "aria-hidden", "true" );
			progress.innerHTML = "<span></span>";
			document.body.appendChild( progress );
		},
		updateReadingProgress: function()
		{
			const progress = document.querySelector( "#" + this.config.progressId + " span" );
			if( !progress )
			{
				return;
			}
			const content = document.querySelector( this.config.contentSelector );
			const scrollTop = window.scrollY || document.documentElement.scrollTop;
			if( content )
			{
				const rect = content.getBoundingClientRect();
				const contentTop = rect.top + scrollTop;
				const contentHeight = Math.max( content.offsetHeight, 1 );
				const contentScroll = Math.min( Math.max( scrollTop - contentTop, 0 ),
						contentHeight );
				const percent = Math.min( Math.max( contentScroll / contentHeight, 0 ), 1 );
				progress.style.width = ( percent * 100 ).toFixed( 2 ) + "%";
				return;
			}
			let maxScroll = document.documentElement.scrollHeight - window.innerHeight;
			if( maxScroll <= 0 )
			{
				maxScroll = 1;
			}
			progress.style.width =
					Math.min( Math.max( ( scrollTop / maxScroll ) * 100, 0 ), 100 ).toFixed( 2 ) +
					"%";
		},
		addScrollTopButton: function()
		{
			if( document.getElementById( this.config.scrollTopId ) )
			{
				return;
			}
			const button = document.createElement( "button" );
			button.id = this.config.scrollTopId;
			button.type = "button";
			button.className = "leeroy-scroll-top";
			button.setAttribute( "aria-label", "Scroll to top" );
			button.setAttribute( "title", "Scroll to top (Alt+T)" );
			button.innerHTML = "↑";
			document.body.appendChild( button );
		},
		updateScrollTopVisibility: function()
		{
			const button = document.getElementById( this.config.scrollTopId );
			if( !button )
			{
				return;
			}
			if( ( window.scrollY || document.documentElement.scrollTop ) > 420 )
			{
				button.classList.add( "is-visible" );
			}
			else
			{
				button.classList.remove( "is-visible" );
			}
		},
		scrollToTop: function()
		{
			window.scrollTo( {
				top: 0,
				behavior: "smooth"
			} );
		},
		addPageTools: function()
		{
			if( document.getElementById( this.config.pageToolsId ) )
			{
				return;
			}
			const content = document.querySelector( this.config.contentSelector );
			if( !content )
			{
				return;
			}
			const title = content.querySelector( "h1" );
			if( !title )
			{
				return;
			}
			const tools = document.createElement( "div" );
			tools.id = this.config.pageToolsId;
			tools.className = "leeroy-page-tools";
			tools.innerHTML = [
				"<button type=\"button\" data-leeroy-copy-page title=\"Copy page link\" aria-label=\"Copy page link\">Copy link</button>",
				"<button type=\"button\" data-leeroy-print-page title=\"Print page\" aria-label=\"Print page\">Print</button>"
			].join( "" );
			title.insertAdjacentElement( "afterend", tools );
		},
		copyPageLink: function( button )
		{
			this.copyTextToClipboard( window.location.href, button, "Copied", "Copy link" );
		},
		copyCurrentPageToClipboard: function()
		{
			const button = document.querySelector( "[data-leeroy-copy-page]" );
			this.copyTextToClipboard( window.location.href, button, "Copied", "Copy link" );
		},
		addHeadingLinks: function()
		{
			const headings = document.querySelectorAll( this.config.headingSelector );
			headings.forEach( function( heading )
			{
				if( heading.querySelector( "." + LeeroyDocs.config.headingLinkClass ) )
				{
					return;
				}
				const button = document.createElement( "button" );
				button.type = "button";
				button.className = LeeroyDocs.config.headingLinkClass;
				button.setAttribute( "data-leeroy-copy-heading", heading.id );
				button.setAttribute( "aria-label", "Copy link to " + heading.textContent.trim() );
				button.setAttribute( "title", "Copy section link" );
				button.textContent = "§";
				heading.appendChild( button );
			} );
		},
		copyHeadingLink: function( button )
		{
			const id = button.getAttribute( "data-leeroy-copy-heading" );
			if( !id )
			{
				return;
			}
			const url = window.location.origin
					+ window.location.pathname
					+ window.location.search
					+ "#"
					+ encodeURIComponent( id );
			this.copyTextToClipboard( url, button, "Copied", "§" );
		},
		copyTextToClipboard: function( text, button, successText, defaultText )
		{
			const updateButton = function()
			{
				if( !button )
				{
					return;
				}
				const previous = button.textContent;
				button.textContent = successText || "Copied";
				setTimeout( function()
				{
					button.textContent = defaultText || previous;
				}, 1400 );
			};
			if( navigator.clipboard && typeof navigator.clipboard.writeText === "function" )
			{
				navigator.clipboard.writeText( text ).then( updateButton ).catch( function()
				{
					LeeroyDocs.fallbackCopyText( text );
					updateButton();
				} );
				return;
			}
			this.fallbackCopyText( text );
			updateButton();
		},
		fallbackCopyText: function( text )
		{
			const textarea = document.createElement( "textarea" );
			textarea.value = text;
			textarea.setAttribute( "readonly", "readonly" );
			textarea.style.position = "fixed";
			textarea.style.top = "-9999px";
			textarea.style.left = "-9999px";
			document.body.appendChild( textarea );
			textarea.select();
			try
			{
				document.execCommand( "copy" );
			}
			catch( error )
			{
				return;
			}
			finally
			{
				document.body.removeChild( textarea );
			}
		},
		addTableFilters: function()
		{
			const tables = document.querySelectorAll( this.config.tableSelector );
			tables.forEach( function( table, index )
			{
				if( table.getAttribute( "data-leeroy-filtered" ) === "true" )
				{
					return;
				}
				const tbody = table.querySelector( "tbody" );
				if( !tbody )
				{
					return;
				}
				const rows = Array.prototype.slice.call( tbody.querySelectorAll( "tr" ) );
				if( rows.length < LeeroyDocs.config.largeTableMinimumRows )
				{
					return;
				}
				table.setAttribute( "data-leeroy-filtered", "true" );
				const wrapper = document.createElement( "div" );
				wrapper.className = "leeroy-table-tools";
				const input = document.createElement( "input" );
				input.type = "search";
				input.className = LeeroyDocs.config.tableFilterClass;
				input.placeholder = "Filter table...";
				input.setAttribute( "aria-label", "Filter table " + ( index + 1 ) );
				const count = document.createElement( "span" );
				count.className = "leeroy-table-count";
				count.textContent = rows.length + " rows";
				wrapper.appendChild( input );
				wrapper.appendChild( count );
				table.parentNode.insertBefore( wrapper, table );
				input.addEventListener( "input", function()
				{
					LeeroyDocs.filterTable( table, input.value, count );
				} );
			} );
		},
		filterTable: function( table, query, countElement )
		{
			const normalizedQuery = ( query || "" ).toLowerCase().trim();
			const rows = Array.prototype.slice.call( table.querySelectorAll( "tbody tr" ) );
			let visible = 0;
			rows.forEach( function( row )
			{
				const text = row.textContent.toLowerCase();
				if( !normalizedQuery || text.indexOf( normalizedQuery ) !== -1 )
				{
					row.style.display = "";
					visible += 1;
				}
				else
				{
					row.style.display = "none";
				}
			} );
			if( countElement )
			{
				countElement.textContent = visible + " / " + rows.length + " rows";
			}
		},
		addCodeLabels: function()
		{
			const codeBlocks = document.querySelectorAll( this.config.codeSelector );
			codeBlocks.forEach( function( code )
			{
				const pre = code.parentElement;
				if( !pre || pre.getAttribute( "data-leeroy-labeled" ) === "true" )
				{
					return;
				}
				const language = LeeroyDocs.detectCodeLanguage( code );
				if( !language )
				{
					return;
				}
				pre.setAttribute( "data-leeroy-labeled", "true" );
				const label = document.createElement( "div" );
				label.className = LeeroyDocs.config.codeLabelClass;
				label.textContent = language;
				pre.insertAdjacentElement( "beforebegin", label );
			} );
		},
		detectCodeLanguage: function( code )
		{
			const className = code.className || "";
			const match = className.match( /language-([a-zA-Z0-9_+-]+)/ );
			if( match && match[ 1 ] )
			{
				return this.formatLanguageName( match[ 1 ] );
			}
			const text = code.textContent.trim();
			if( /^site_name:|^theme:|^plugins:|^nav:/m.test( text ) )
			{
				return "YAML";
			}
			if( /def\s+\w+\(|class\s+\w+/.test( text ) )
			{
				return "Python";
			}
			if( /^mkdocs\s|^python\s|-m\s+|^streamlit\s|^pip\s|^git\s/m.test( text ) )
			{
				return "Shell";
			}
			if( /^\{[\s\S]*\}$/.test( text ) )
			{
				return "JSON";
			}
			if( /^#\s|^##\s|```/.test( text ) )
			{
				return "Markdown";
			}
			if( /SELECT\s+|CREATE\s+TABLE|INSERT\s+INTO|PRAGMA\s+/i.test( text ) )
			{
				return "SQL";
			}
			return "";
		},
		formatLanguageName: function( language )
		{
			const map = {
				py: "Python",
				python: "Python",
				ps1: "PowerShell",
				powershell: "PowerShell",
				bash: "Shell",
				sh: "Shell",
				shell: "Shell",
				yaml: "YAML",
				yml: "YAML",
				json: "JSON",
				md: "Markdown",
				markdown: "Markdown",
				html: "HTML",
				css: "CSS",
				js: "JavaScript",
				javascript: "JavaScript",
				sql: "SQL",
				text: "Text",
				txt: "Text"
			};
			const key = String( language || "" ).toLowerCase();
			return map[ key ] || key.toUpperCase();
		},
		addCodeToggles: function()
		{
			const codeBlocks = document.querySelectorAll( this.config.codeSelector );
			codeBlocks.forEach( function( code )
			{
				const pre = code.parentElement;
				if( !pre || pre.getAttribute( "data-leeroy-toggle-ready" ) === "true" )
				{
					return;
				}
				pre.setAttribute( "data-leeroy-toggle-ready", "true" );
				if( pre.scrollHeight <= LeeroyDocs.config.maxCollapsedCodeHeight + 80 )
				{
					return;
				}
				pre.classList.add( "leeroy-code-collapsed" );
				pre.style.maxHeight = LeeroyDocs.config.maxCollapsedCodeHeight + "px";
				const button = document.createElement( "button" );
				button.type = "button";
				button.className = LeeroyDocs.config.codeToggleClass;
				button.setAttribute( "data-leeroy-toggle-code", "collapsed" );
				button.textContent = "Show full code";
				pre.insertAdjacentElement( "afterend", button );
			} );
		},
		toggleCodeBlock: function( button )
		{
			const pre = button.previousElementSibling;
			if( !pre || pre.tagName.toLowerCase() !== "pre" )
			{
				return;
			}
			const state = button.getAttribute( "data-leeroy-toggle-code" );
			if( state === "collapsed" )
			{
				pre.classList.remove( "leeroy-code-collapsed" );
				pre.style.maxHeight = "";
				button.setAttribute( "data-leeroy-toggle-code", "expanded" );
				button.textContent = "Collapse code";
			}
			else
			{
				pre.classList.add( "leeroy-code-collapsed" );
				pre.style.maxHeight = this.config.maxCollapsedCodeHeight + "px";
				button.setAttribute( "data-leeroy-toggle-code", "collapsed" );
				button.textContent = "Show full code";
			}
		},
		addPagePathMetadata: function()
		{
			const content = document.querySelector( this.config.contentSelector );
			if( !content || content.querySelector( ".leeroy-page-path" ) )
			{
				return;
			}
			const h1 = content.querySelector( "h1" );
			if( !h1 )
			{
				return;
			}
			const path = window.location.pathname
					.replace( /\/$/, "" )
					.split( "/" )
					.filter( Boolean )
					.slice( -4 )
					.join( " / " );
			if( !path )
			{
				return;
			}
			const meta = document.createElement( "div" );
			meta.className = "leeroy-page-path";
			meta.textContent = "Docs path: " + path;
			h1.insertAdjacentElement( "afterend", meta );
		},
		saveNavigationScroll: function()
		{
			const nav = document.querySelector( this.config.navSelector );
			if( !nav )
			{
				return;
			}
			try
			{
				window.sessionStorage.setItem( this.config.navScrollKey,
						String( nav.scrollTop || 0 ) );
			}
			catch( error )
			{
				return;
			}
		},
		restoreNavigationScroll: function()
		{
			const nav = document.querySelector( this.config.navSelector );
			if( !nav )
			{
				return;
			}
			try
			{
				const value = window.sessionStorage.getItem( this.config.navScrollKey );
				if( value !== null )
				{
					nav.scrollTop = parseInt( value, 10 ) || 0;
				}
			}
			catch( error )
			{
				return;
			}
		},
		enhanceKeyboardFocus: function()
		{
			document.body.addEventListener( "keydown", function( event )
			{
				if( event.key === "Tab" )
				{
					document.body.classList.add( "leeroy-keyboard-mode" );
				}
			} );
			document.body.addEventListener( "mousedown", function()
			{
				document.body.classList.remove( "leeroy-keyboard-mode" );
			} );
		},
		enhanceApiReference: function()
		{
			const apiContainers = document.querySelectorAll( this.config.apiObjectSelector );
			apiContainers.forEach( function( container )
			{
				if( container.getAttribute( "data-leeroy-api-enhanced" ) === "true" )
				{
					return;
				}
				container.setAttribute( "data-leeroy-api-enhanced", "true" );
				const heading = container.querySelector( "h2, h3, h4, h5" );
				if( !heading || heading.querySelector( ".leeroy-api-badge" ) )
				{
					return;
				}
				const badge = document.createElement( "span" );
				badge.className = "leeroy-api-badge";
				if( container.className.indexOf( "doc-class" ) !== -1 )
				{
					badge.textContent = "class";
				}
				else if( container.className.indexOf( "doc-method" ) !== -1 )
				{
					badge.textContent = "method";
				}
				else if( container.className.indexOf( "doc-function" ) !== -1 )
				{
					badge.textContent = "function";
				}
				else if( container.className.indexOf( "doc-attribute" ) !== -1 )
				{
					badge.textContent = "attribute";
				}
				else if( container.className.indexOf( "doc-property" ) !== -1 )
				{
					badge.textContent = "property";
				}
				else
				{
					badge.textContent = "api";
				}
				heading.appendChild( badge );
			} );
		},
		addApiTools: function()
		{
			const content = document.querySelector( this.config.contentSelector );
			if( !content || content.querySelector( ".leeroy-api-tools" ) )
			{
				return;
			}
			const apiObjects = content.querySelectorAll( this.config.apiObjectSelector );
			const detailsBlocks = content.querySelectorAll( "details" );
			if( apiObjects.length === 0 && detailsBlocks.length === 0 )
			{
				return;
			}
			const firstHeading = content.querySelector( "h1" );
			if( !firstHeading )
			{
				return;
			}
			const panel = document.createElement( "section" );
			panel.className = "leeroy-api-tools";
			panel.setAttribute( "aria-label", "API tools" );
			panel.innerHTML = [
				"<h2 class=\"leeroy-api-tools-title\">API Tools</h2>",
				"<label class=\"leeroy-api-search-label\" for=\"" + this.config.apiSearchId +
				"\">Filter classes, functions, methods, properties, or text</label>",
				"<input id=\"" + this.config.apiSearchId +
				"\" class=\"leeroy-api-search\" type=\"search\" placeholder=\"Filter API reference...\" autocomplete=\"off\">",
				"<div class=\"leeroy-api-tool-buttons\">",
				"<button type=\"button\" class=\"leeroy-api-tool-button\" data-leeroy-api-expand>Expand all</button>",
				"<button type=\"button\" class=\"leeroy-api-tool-button\" data-leeroy-api-collapse>Collapse all</button>",
				"<button type=\"button\" class=\"leeroy-api-tool-button\" data-leeroy-api-clear>Clear filter</button>",
				"</div>",
				"<p class=\"leeroy-api-filter-status\" aria-live=\"polite\"></p>"
			].join( "" );
			firstHeading.insertAdjacentElement( "afterend", panel );
			const input = panel.querySelector( "#" + this.config.apiSearchId );
			if( input )
			{
				input.addEventListener( "input", function()
				{
					LeeroyDocs.filterApiObjects( input.value );
				} );
			}
		},
		filterApiObjects: function( query )
		{
			const content = document.querySelector( this.config.contentSelector );
			if( !content )
			{
				return;
			}
			const normalizedQuery = ( query || "" ).toLowerCase().trim();
			const objects = Array.prototype.slice.call(
					content.querySelectorAll( this.config.apiObjectSelector ) );
			const status = content.querySelector( ".leeroy-api-filter-status" );
			let visible = 0;
			objects.forEach( function( object )
			{
				const text = object.textContent.toLowerCase();
				if( !normalizedQuery || text.indexOf( normalizedQuery ) !== -1 )
				{
					object.classList.remove( "leeroy-api-hidden" );
					visible += 1;
				}
				else
				{
					object.classList.add( "leeroy-api-hidden" );
				}
			} );
			if( status )
			{
				if( !objects.length )
				{
					status.textContent = "No API objects detected.";
				}
				else if( !normalizedQuery )
				{
					status.textContent = objects.length + " API objects";
				}
				else
				{
					status.textContent = visible + " / " + objects.length + " API objects";
				}
			}
		},
		clearApiFilter: function()
		{
			const input = document.getElementById( this.config.apiSearchId );
			if( input )
			{
				input.value = "";
			}
			this.filterApiObjects( "" );
		},
		setApiDetailsState: function( expanded )
		{
			const content = document.querySelector( this.config.contentSelector );
			if( !content )
			{
				return;
			}
			const details = content.querySelectorAll( "details" );
			details.forEach( function( item )
			{
				item.open = Boolean( expanded );
			} );
		},
		enhanceTocProgress: function()
		{
			const toc = document.querySelector( this.config.tocSelector );
			if( !toc )
			{
				return;
			}
			const links = toc.querySelectorAll( "a[href^='#']" );
			links.forEach( function( link )
			{
				if( !link.querySelector( ".leeroy-toc-marker" ) )
				{
					const marker = document.createElement( "span" );
					marker.className = "leeroy-toc-marker";
					marker.setAttribute( "aria-hidden", "true" );
					link.prepend( marker );
				}
			} );
		},
		updateTocProgress: function()
		{
			const headings = Array.prototype.slice.call(
					document.querySelectorAll( this.config.headingSelector ) );
			if( !headings.length )
			{
				return;
			}
			let activeId = "";
			for( let i = 0; i < headings.length; i += 1 )
			{
				const rect = headings[ i ].getBoundingClientRect();
				if( rect.top <= 120 )
				{
					activeId = headings[ i ].id;
				}
				else
				{
					break;
				}
			}
			if( !activeId && headings[ 0 ] )
			{
				activeId = headings[ 0 ].id;
			}
			const tocLinks = document.querySelectorAll( this.config.tocSelector + " a[href^='#']" );
			tocLinks.forEach( function( link )
			{
				const href = link.getAttribute( "href" ) || "";
				const targetId = decodeURIComponent( href.replace( /^#/, "" ) );
				if( targetId === activeId )
				{
					link.classList.add( "leeroy-toc-active" );
				}
				else
				{
					link.classList.remove( "leeroy-toc-active" );
				}
			} );
		},
		addMermaidGuard: function()
		{
			const mermaidBlocks = document.querySelectorAll( ".mermaid" );
			mermaidBlocks.forEach( function( block )
			{
				if( block.getAttribute( "data-leeroy-mermaid-guard" ) === "true" )
				{
					return;
				}
				block.setAttribute( "data-leeroy-mermaid-guard", "true" );
				block.setAttribute( "role", "img" );
				if( !block.getAttribute( "aria-label" ) )
				{
					block.setAttribute( "aria-label", "Diagram" );
				}
			} );
		}
	};
	
	function initializeLeeroyDocs()
	{
		try
		{
			LeeroyDocs.init();
		}
		catch( error )
		{
			return;
		}
	}
	
	if( document.readyState === "loading" )
	{
		document.addEventListener( "DOMContentLoaded", initializeLeeroyDocs );
	}
	else
	{
		initializeLeeroyDocs();
	}
} )();